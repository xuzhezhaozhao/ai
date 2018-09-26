#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import os
import threading
import time
import Queue

from datetime import datetime
from tensorflow.python.eager import context
from tensorflow.python.util import function_utils


class EasyEstimator(tf.estimator.Estimator):
    def __init__(self, model_fn, model_dir=None, config=None, params=None,
                 warm_start_from=None, num_parallel=1, log_step_count_secs=10):

        super(EasyEstimator, self).__init__(
            model_fn=model_fn, model_dir=model_dir, config=config,
            params=params, warm_start_from=warm_start_from)

        self._num_parallel = num_parallel
        self._log_step_count_secs = log_step_count_secs

    def easy_train(self, input_fn, steps=None, max_steps=None,
                   evaluate_every_secs=None, evaluate_input_fn=None,
                   evaluate_steps=None):
        with context.graph_mode():
            if (steps is not None) and (max_steps is not None):
                raise ValueError(
                    'Can not provide both steps and max_steps.')
            if steps is not None and steps <= 0:
                raise ValueError(
                    'Must specify steps > 0, given: {}'.format(steps))
            if max_steps is not None and max_steps <= 0:
                raise ValueError('Must specify max_steps > 0, given: {}'
                                 .format(max_steps))

            if max_steps is not None:
                start_step = _load_global_step_from_checkpoint_dir(
                    self._model_dir)
                if max_steps <= start_step:
                    tf.logging.info('Skipping training since max_steps '
                                    'has already saved.\n')
                    return self

            loss = self._easy_train_model(input_fn, steps, max_steps,
                                          evaluate_every_secs,
                                          evaluate_input_fn, evaluate_steps)
            tf.logging.info('Loss for final step: %s.\n', loss)
            return self

    def _easy_train_model(self, input_fn, steps, max_steps,
                          evaluate_every_secs, evaluate_input_fn,
                          evaluate_steps):
        with tf.Graph().as_default() as g, g.device(self._device_fn):
            tf.set_random_seed(self._config.tf_random_seed)
            global_step_tensor = tf.train.get_or_create_global_step(g)

            self._iterator_initializers = []
            (features, labels) = \
                self._easy_get_features_and_labels_from_input_fn(
                    input_fn, tf.estimator.ModeKeys.TRAIN)
            estimator_spec = self._easy_call_model_fn(
                features, labels, tf.estimator.ModeKeys.TRAIN, self.config)

            merged_summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self._model_dir)
            saver = tf.train.Saver(
                sharded=True,
                max_to_keep=self._config.keep_checkpoint_max,
                save_relative_paths=True)

            with tf.Session() as sess:
                self._session_init(sess, summary_writer, saver)

                tf.logging.info('{} Start training ...'.format(datetime.now()))
                queue = Queue.Queue()
                init_steps = sess.run(global_step_tensor)
                workers = self._start_train_threads(
                    sess, estimator_spec, queue)

                current_time = time.time()
                self._last_save_summary_steps = init_steps
                self._last_save_checkpoint_time = current_time
                self._last_evaluate_time = current_time

                while True:
                    time.sleep(self._log_step_count_secs)
                    current_time = time.time()
                    loss, global_steps, end = self._run_one_step(
                        sess, estimator_spec, global_step_tensor)
                    if end:
                        break
                    self._loss_logging(sess, global_steps, loss)
                    self._should_stop = self._check_should_stop(
                        init_steps, global_steps, steps, max_steps)
                    if self._should_stop:
                        break

                    self._maybe_save_summary(
                        sess, summary_writer, merged_summary, global_steps)
                    self._maybe_save_checkpoint(
                        sess, current_time, global_steps, saver)
                    self._maybe_evaluate(
                        sess, evaluate_every_secs, current_time, global_steps,
                        saver, evaluate_input_fn, evaluate_steps)

                loss = self._wait_train_threads(workers, queue)
                self._close_train(sess, global_steps, loss, saver,
                                  summary_writer)

            return loss

    def _easy_get_features_and_labels_from_input_fn(self, input_fn, mode):
        result = self._easy_call_input_fn(input_fn, mode)

        return self._easy_parse_input_fn_result(result)

    def _easy_call_input_fn(self, input_fn, mode):
        with tf.device('/cpu:0'):
            return input_fn()

    def _easy_call_model_fn(self, features, labels, mode, config):
        model_fn_args = function_utils.fn_args(self._model_fn)
        kwargs = {}
        if 'labels' in model_fn_args:
            kwargs['labels'] = labels
        else:
            if labels is not None:
                raise ValueError('model_fn does not take labels, '
                                 'but input_fn returns labels.')
        if 'mode' in model_fn_args:
            kwargs['mode'] = mode
        if 'params' in model_fn_args:
            kwargs['params'] = self.params
        if 'config' in model_fn_args:
            kwargs['config'] = config

        tf.logging.info('Calling model_fn.')
        model_fn_results = self._model_fn(features=features, **kwargs)
        tf.logging.info('Done calling model_fn.\n')

        if not isinstance(model_fn_results, tf.estimator.EstimatorSpec):
            raise ValueError('model_fn should return an EstimatorSpec.')

        return model_fn_results

    def _easy_parse_input_fn_result(self, result):
        iterator = result.make_initializable_iterator()
        self._iterator_initializers.append(iterator.initializer)
        data = iterator.get_next()

        if isinstance(data, (list, tuple)):
            if len(data) != 2:
                raise ValueError('input_fn should return (features, labels)'
                                 ' as a len 2 tuple.')
            return data[0], data[1]
        return data, None

    def _easy_maybe_restore_model(self, sess, saver):
        lastest_path = tf.train.latest_checkpoint(self._model_dir)
        if lastest_path is not None:
            tf.logging.info('restore model ckpt from {} ...'
                            .format(lastest_path))
            saver.restore(sess, lastest_path)

    def _loss_logging(self, sess, global_steps, loss):
        tf.logging.info('{} global_steps = {}, loss = {}'
                        .format(datetime.now(), global_steps, loss))

    def _save_checkpoint(self, sess, global_steps, saver):
        ckpt_name = os.path.join(self._model_dir,
                                 'model.ckpt-{}'.format(global_steps))
        ckpt_path = saver.save(sess, ckpt_name)
        tf.logging.info('{} Model ckpt saved at {}'
                        .format(datetime.now(), ckpt_path))

    def _save_summary(self, sess, global_steps, writer, merged_summary):
        merged_summary = sess.run(merged_summary)
        writer.add_summary(merged_summary, global_steps)
        writer.flush()
        tf.logging.info('{} Summary saved.'.format(datetime.now()))

    def _check_global_steps(self, global_steps, steps):
        return True if global_steps % steps == 0 else False

    def _session_init(self, sess, summary_writer, saver):
        sess.run(tf.global_variables_initializer())
        sess.run(self._iterator_initializers)
        summary_writer.add_graph(sess.graph)
        tf.get_default_graph().finalize()
        self._easy_maybe_restore_model(sess, saver)

    def _run_one_step(self, sess, estimator_spec, global_step_tensor):
        try:
            _, loss = sess.run([estimator_spec.train_op, estimator_spec.loss])
            global_steps = sess.run(global_step_tensor)
        except tf.errors.OutOfRangeError:
            return 0, 0, True
        return loss, global_steps, False

    def _check_should_stop(self, init_steps, global_steps, steps, max_steps):
        if max_steps and (global_steps >= max_steps):
            return True
        if steps and (global_steps - init_steps >= steps):
            return True
        return False

    def _maybe_save_summary(self, sess, summary_writer, merged_summary,
                            current_steps):
        last_steps = self._last_save_summary_steps
        if current_steps - last_steps >= self._config.save_summary_steps:
            self._save_summary(
                sess, current_steps, summary_writer, merged_summary)
            self._last_save_summary_steps = current_steps

    def _maybe_save_checkpoint(self, sess, current_time, current_steps, saver):
        last_time = self._last_save_checkpoint_time
        if current_time - last_time >= self._config.save_checkpoints_secs:
            self._save_checkpoint(sess, current_steps, saver)
            self._last_save_checkpoint_time = current_time

    def _maybe_evaluate(self, sess, evaluate_every_secs, current_time,
                        global_steps, saver, evaluate_input_fn,
                        evaluate_steps):
        last_evaluation_time = self._last_evaluate_time
        if (evaluate_every_secs and (current_time - last_evaluation_time
                                     > evaluate_every_secs)):
            self._save_checkpoint(sess, global_steps, saver)
            self._wait_evaluation = True
            tf.logging.info('\n{} Starting evaluation [in training] ...'
                            .format(datetime.now()))
            self.evaluate(
                input_fn=evaluate_input_fn,
                steps=evaluate_steps)
            tf.logging.info('{} Evaluation OK [in training].\n'
                            .format(datetime.now()))
            self._last_evaluate_time = time.time()  # evaluate took time
            self._wait_evaluation = False

    def _start_train_threads(self, sess, estimator_spec, queue):
        self._should_stop = False
        self._wait_evaluation = False
        workers = []
        for tid in range(self._num_parallel):
            tf.logging.info('Start thread {} ...'.format(tid))
            worker = threading.Thread(
                target=self._train_thread_body,
                args=(tid, sess, estimator_spec, queue))
            worker.start()
            workers.append(worker)
        return workers

    def _wait_train_threads(self, workers, queue):
        for worker in workers:
            worker.join()

        cnt = 0
        loss = 0.0
        while not queue.empty():
            single_loss = queue.get()
            tf.logging.info('thread loss: {}'.format(single_loss))
            if single_loss is None:
                continue
            loss += single_loss
            cnt += 1

        loss = (loss / cnt if cnt != 0 else None)

        return loss

    def _train_thread_body(self, tid, sess, estimator_spec, queue):
        loss = None
        while not self._should_stop:
            try:
                while self._wait_evaluation:
                    tf.logging.info('{} thread {} wait evaluation ...'
                                    .format(datetime.now(), tid))
                    time.sleep(60)
                _, loss = sess.run(
                    [estimator_spec.train_op, estimator_spec.loss])
            except tf.errors.OutOfRangeError:
                tf.logging.info(
                    "thread {} catch 'OutOfRangeError'".format(tid))
                break

        queue.put(loss)
        tf.logging.info('thread {} exit.'.format(tid))
        return loss

    def _close_train(self, sess, global_steps, loss, saver, summary_writer):
        self._loss_logging(sess, global_steps, loss)
        self._save_checkpoint(sess, global_steps, saver)
        summary_writer.close()


def _load_global_step_from_checkpoint_dir(checkpoint_dir):
    try:
        checkpoint_reader = tf.train.NewCheckpointReader(
            tf.train.latest_checkpoint(checkpoint_dir))
        return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
    except Exception:
        return 0


def _check_stop(current_steps, global_steps, steps, max_steps):
    if steps is not None and current_steps >= steps:
        return True
    if max_steps is not None and global_steps >= max_steps:
        return True
    return False
