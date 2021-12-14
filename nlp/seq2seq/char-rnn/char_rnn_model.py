#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import tensorflow as tf
import numpy as np
import os
import time

import input_data


class CharRNN(object):
    def __init__(self, opts, params):
        self.opts = opts
        self.params = params
        self.vocab_size = params['vocab_size']
        tf.logging.info("vocab_size = {}".format(self.vocab_size))

        num_samples_per_epoch = self.params['num_samples_per_epoch']
        num_steps_per_epoch = int(num_samples_per_epoch/self.opts.batch_size)
        self.train_steps = num_steps_per_epoch * self.opts.epoch
        tf.logging.info("train_steps = {}".format(self.train_steps))

        self.inputs = None
        self.labels = None
        self.iterator_initializer = None
        self.initial_state = None
        self.final_state = None
        self.logits = None
        self.probs = None

    def build_graph(self, training):
        """Build char rnn model graph."""

        tf.summary.histogram('inputs', self.inputs)

        # use embedding for Chinese, not necessary for English
        if not self.opts.use_embedding:
            lstm_inputs = tf.one_hot(self.inputs, self.vocab_size)
        else:
            embedding = tf.get_variable(
                'embedding', [self.vocab_size, self.opts.embedding_dim])
            lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

        if training:
            keep_prob = self.opts.keep_prob
        else:
            keep_prob = 1.0

        cell = tf.nn.rnn_cell.MultiRNNCell(
            [self.get_rnn_cell(self.opts.hidden_size, keep_prob)
             for _ in range(self.opts.num_layers)]
        )
        if training:
            batch_size = self.opts.batch_size
        else:
            batch_size = 1
        # batch_size = self.inputs.shape.as_list()[0]
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # lstm_outputs: shape [batch, seq_length, hidden_size]
        # states：states表示最终的状态，也就是序列中最后一个cell输出的状态。
        # 一般情况下states的形状为 [batch_size, cell.output_size]，但当输入的
        # cell为BasicLSTMCell时，state的形状为[2，batch_size, cell.output_size]，
        # 其中2也对应着LSTM中的cell state和hidden state。
        # final_state 是个 N-tuple，tuple里面的shape是 (2, batch_size, output_size)
        # 打印出来是这样： final_state shape:(LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_3:0' shape=(32, 128) dtype=float32>, h=<tf.Tensor'rnn/while/Exit_4:0' shape=(32, 128) dtype=float32>), LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_5:0' shape=(32, 128) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_6:0' shape=(32, 128) dtype=float32>))
        lstm_outputs, self.final_state = tf.nn.dynamic_rnn(
            cell, lstm_inputs, initial_state=self.initial_state)
        print("final_state shape:{}".format(self.final_state))

        # shape [batch, seq_length, vocab_size]
        self.logits = tf.layers.dense(lstm_outputs, self.vocab_size)
        self.probs = tf.nn.softmax(self.logits)

    def train(self, input_fn):
        with tf.Graph().as_default() as g:
            # tf.set_random_seed(None)

            result = self.call_input_fn(input_fn)
            self.inputs = tf.placeholder(
                tf.int64, [self.opts.batch_size, self.opts.seq_length])
            self.labels = tf.placeholder(
                tf.int64, [self.opts.batch_size, self.opts.seq_length])
            global_step_tensor = tf.train.get_or_create_global_step(g)
            self.build_graph(training=True)
            train_op, loss_tensor = self.create_train_op_and_loss(
                self.logits, self.labels)

            with tf.Session() as sess:
                (merged_summary, writer,
                 saver) = self.create_writer_and_saver(sess)
                self.session_init(sess, saver)
                tf.logging.info('{} Start training ...'.format(self.now()))
                final_state = sess.run(self.initial_state)
                for _ in xrange(self.train_steps):
                    inputs, labels = result.next()
                    feed_dict = {
                        self.initial_state: final_state,
                        self.inputs: inputs,
                        self.labels: labels
                    }
                    run_ops = [train_op, loss_tensor,
                               self.final_state, global_step_tensor]

                    start_time = time.time()
                    _, loss, final_state, global_step = sess.run(
                        run_ops, feed_dict=feed_dict)
                    duration_sec = time.time() - start_time

                    self.maybe_logging(global_step, loss, duration_sec)
                    self.maybe_save_checkpoint(sess, saver, global_step)
                    self.maybe_save_summary(
                        sess, writer, merged_summary, global_step, feed_dict)

                # save for last step
                self.save_checkpoint(sess, saver, global_step)

    def sample(self):
        with tf.Graph().as_default():
            start_string = self.opts.start_string.decode('utf-8')
            text_as_int = input_data.text_to_int(start_string, self.opts)
            self.inputs = tf.placeholder(tf.int64, shape=(1, None))
            self.build_graph(training=False)

            with tf.Session() as sess:
                (merged_summary, writer,
                 saver) = self.create_writer_and_saver(sess)
                self.session_init(sess, saver)

                tf.logging.info('{} Start sampling ...'.format(self.now()))
                inputs = text_as_int.reshape([1, len(text_as_int)])
                samples = text_as_int.tolist()
                final_state = sess.run(self.initial_state)
                for _ in range(self.opts.num_samples):
                    feed_dict = {
                        self.inputs: inputs,
                        self.initial_state: final_state
                    }
                    probs, final_state = sess.run(
                        [self.probs, self.final_state],
                        feed_dict=feed_dict)
                    if probs.shape[1] == 0:
                        probs = np.ones((1, 1, self.vocab_size))
                        k = self.vocab_size
                    else:
                        k = self.opts.sample_top_k

                    c = self.pick_top_k(probs, k)
                    samples.append(c)
                    inputs = np.array([[c]])

                samples = input_data.idx_to_text(samples, self.opts)
                samples = ''.join(samples).encode('utf-8')
                tf.logging.info("sampels: \n{}".format(samples))
                tf.logging.info('{} Sampling done.'.format(self.now()))

    def create_writer_and_saver(self, sess):
        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.opts.model_dir)
        summary_writer.add_graph(sess.graph)
        saver = tf.train.Saver(
            sharded=True,
            max_to_keep=self.opts.keep_checkpoint_max,
            save_relative_paths=True)
        return merged_summary, summary_writer, saver

    def session_init(self, sess, saver):
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if self.iterator_initializer is not None:
            sess.run(self.iterator_initializer)
        tf.get_default_graph().finalize()
        self.maybe_restore_model(sess, saver)

    def maybe_restore_model(self, sess, saver):
        lastest_path = tf.train.latest_checkpoint(self.opts.model_dir)
        if lastest_path is not None:
            tf.logging.info('restore model ckpt from {} ...'
                            .format(lastest_path))
            saver.restore(sess, lastest_path)

    def maybe_save_checkpoint(self, sess, saver, global_step):
        if global_step % self.opts.save_checkpoints_steps == 0:
            self.save_checkpoint(sess, saver, global_step)

    def save_checkpoint(self, sess, saver, global_step):
        ckpt_name = os.path.join(
            self.opts.model_dir, 'model.ckpt-{}'.format(global_step))
        ckpt_path = saver.save(sess, ckpt_name)
        tf.logging.info('{} Model ckpt saved at "{}"'
                        .format(self.now(), ckpt_path))

    def maybe_logging(self, global_step, loss, duration_sec):
        if (global_step == 1 or
                global_step % self.opts.log_step_count_steps == 0):
            tf.logging.info("step = {}, loss = {}, time = {:.3f} s/step"
                            .format(global_step, loss, duration_sec))

    def maybe_save_summary(self, sess, writer, merged_summary, global_step,
                           feed_dict):
        if global_step % self.opts.save_summary_steps == 0:
            self.save_summary(
                sess, writer, merged_summary, global_step, feed_dict)

    def save_summary(self, sess, writer, merged_summary, global_step,
                     feed_dict):
            merged_summary = sess.run(merged_summary, feed_dict=feed_dict)
            writer.add_summary(merged_summary, global_step)
            writer.flush()

    def call_input_fn(self, input_fn):
        with tf.device('/cpu:0'):
            return input_fn()

    def get_rnn_cell(self, size, keep_prob):
        lstm = tf.nn.rnn_cell.LSTMCell(size)
        lstm = tf.nn.rnn_cell.DropoutWrapper(
            lstm,
            output_keep_prob=keep_prob,
            variational_recurrent=True,
            dtype=tf.float32)
        return lstm

    def get_loss(self, logits, labels):
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)
        return loss

    def create_train_op_and_loss(self, logits, labels):

        num_samples_per_epoch = self.params['num_samples_per_epoch']
        loss = self.get_loss(logits, labels)
        tf.summary.scalar('loss', loss)

        global_step = tf.train.get_global_step()
        learning_rate = self.configure_learning_rate(
            num_samples_per_epoch, global_step)
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = self.configure_optimizer(learning_rate)

        gradients, variables = zip(*optimizer.compute_gradients(loss))
        if self.opts.use_clip_gradients:
            gradients, _ = tf.clip_by_global_norm(gradients,
                                                  self.opts.clip_norm)

        train_op = optimizer.apply_gradients(
            zip(gradients, variables),
            global_step=global_step)

        for var, grad in zip(variables, gradients):
            tf.summary.histogram(
                var.name.replace(':', '_') + '/gradient', grad)

        return train_op, loss

    def configure_optimizer(self, learning_rate):
        """Configures the optimizer used for training."""

        if self.opts.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(
                learning_rate,
                rho=self.opts.adadelta_rho,
                epsilon=self.opts.opt_epsilon)
        elif self.opts.optimizer == 'adagrad':
            accumulator_value = self.opts.adagrad_initial_accumulator_value
            optimizer = tf.train.AdagradOptimizer(
                learning_rate,
                initial_accumulator_value=accumulator_value)
        elif self.opts.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate,
                beta1=self.opts.adam_beta1,
                beta2=self.opts.adam_beta2,
                epsilon=self.opts.opt_epsilon)
        elif self.opts.optimizer == 'ftrl':
            accumulator_value = self.opts.ftrl_initial_accumulator_value
            optimizer = tf.train.FtrlOptimizer(
                learning_rate,
                learning_rate_power=self.opts.ftrl_learning_rate_power,
                initial_accumulator_value=accumulator_value,
                l1_regularization_strength=self.opts.ftrl_l1,
                l2_regularization_strength=self.opts.ftrl_l2)
        elif self.opts.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate,
                momentum=self.opts.momentum,
                name='Momentum')
        elif self.opts.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate,
                decay=self.opts.rmsprop_decay,
                momentum=self.opts.rmsprop_momentum,
                epsilon=self.opts.opt_epsilon)
        elif self.opts.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('Optimizer [%s] was not recognized'
                             % self.opts.optimizer)
        return optimizer

    def configure_learning_rate(self, num_samples_per_epoch, global_step):
        """Configures the learning rate."""

        decay_steps = int(num_samples_per_epoch *
                          self.opts.num_epochs_per_decay /
                          self.opts.batch_size)

        tf.logging.info('decay_steps = {}'.format(decay_steps))
        if self.opts.learning_rate_decay_type == 'exponential':
            return tf.train.exponential_decay(
                self.opts.learning_rate,
                global_step,
                decay_steps,
                self.opts.learning_rate_decay_factor,
                staircase=True,
                name='exponential_decay_learning_rate')
        elif self.opts.learning_rate_decay_type == 'fixed':
            return tf.constant(self.opts.learning_rate,
                               name='fixed_learning_rate')
        elif self.opts.learning_rate_decay_type == 'polynomial':
            return tf.train.polynomial_decay(
                self.opts.learning_rate,
                global_step,
                decay_steps,
                self.opts.end_learning_rate,
                power=1.0,
                cycle=False,
                name='polynomial_decay_learning_rate')
        else:
            raise ValueError(
                'learning_rate_decay_type [%s] was not recognized' %
                self.opts.learning_rate_decay_type)

    def now(self):
        return datetime.now()

    def pick_top_k(self, probs, top_k=5):
        probs = probs[-1, -1, :]
        probs[np.argsort(probs)[:-top_k]] = 0
        probs = probs / np.sum(probs)  # normalize
        c = np.random.choice(self.vocab_size, 1, p=probs)[0]
        return c
