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
                   evaluation_every_secs=None):
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
                                    'has already saved.')
                    return self

            loss = self._easy_train_model(input_fn, steps, max_steps)
            tf.logging.info('Loss for final step: %s.', loss)
            return self

    def _easy_train_model(self, input_fn, steps, max_steps):
        with tf.Graph().as_default() as g, g.device(self._device_fn):
            tf.set_random_seed(self._config.tf_random_seed)
            global_step_tensor = tf.train.get_or_create_global_step(g)
            (features,
             labels) = self._easy_get_features_and_labels_from_input_fn(
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
                sess.run(tf.global_variables_initializer())
                sess.run(self._iterator_initializer)
                summary_writer.add_graph(sess.graph)

                tf.get_default_graph().finalize()

                self._easy_maybe_restore_model(sess, saver)

                tf.logging.info('{} Start training ...'.format(datetime.now()))

                workers = []
                queue = Queue.Queue()
                init_steps = sess.run(global_step_tensor)
                self._should_stop = False

                for tid in range(self._num_parallel):
                    tf.logging.info('Start thread {} ...'.format(tid))
                    worker = threading.Thread(
                        target=self._train_thread_body,
                        args=(tid, sess, estimator_spec, queue))
                    worker.start()
                    workers.append(worker)

                last_steps = init_steps
                last_time = time.time()
                while True:
                    time.sleep(self._log_step_count_secs)
                    try:
                        _, loss = sess.run([estimator_spec.train_op,
                                            estimator_spec.loss])
                        global_steps = sess.run(global_step_tensor)
                        current_time = time.time()

                        self._loss_logging(sess, global_steps, loss)

                        if max_steps and (global_steps >= max_steps):
                            self._should_stop = True
                        if steps and (global_steps - init_steps >= steps):
                            self._should_stop = True

                        if self._should_stop:
                            break

                        if (global_steps - last_steps
                                >= self._config.save_summary_steps):
                            self._save_summary(
                                sess, global_steps,
                                summary_writer, merged_summary)

                        if (current_time - last_time
                                >= self._config.save_checkpoints_secs):
                            self._save_ckpt(sess, global_steps, saver)

                        last_steps = global_steps
                        last_time = current_time
                    except tf.errors.OutOfRangeError:
                        break

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

                self._loss_logging(sess, global_steps, loss)
                self._save_checkpoint(sess, global_steps, saver)
                summary_writer.close()

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
        tf.logging.info('Done calling model_fn.')

        if not isinstance(model_fn_results, tf.estimator.EstimatorSpec):
            raise ValueError('model_fn should return an EstimatorSpec.')

        return model_fn_results

    def _easy_parse_input_fn_result(self, result):
        iterator = result.make_initializable_iterator()
        self._iterator_initializer = iterator.initializer
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

    def _train_thread_body(self, tid, sess, estimator_spec, queue):
        loss = None
        while not self._should_stop:
            try:
                _, loss = sess.run(
                    [estimator_spec.train_op, estimator_spec.loss])
            except tf.errors.OutOfRangeError:
                tf.logging.info(
                    "thread {} catch 'OutOfRangeError'".format(tid))
                break

        queue.put(loss)
        tf.logging.info('thread {} exit.'.format(tid))
        return loss


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
