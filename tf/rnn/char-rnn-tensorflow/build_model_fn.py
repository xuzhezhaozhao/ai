#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def model_fn(features, labels, mode, params):

    opts = params['opts']
    vocab_size = params['vocab_size']

    # use embedding for Chinese, not necessary for English
    if not opts.use_embedding:
        lstm_inputs = tf.one_hot(features, vocab_size)
    else:
        embedding = tf.get_variable('embedding',
                                    [vocab_size, opts.embedding_size])
        lstm_inputs = tf.nn.embedding_lookup(embedding, features)

    cell = tf.nn.rnn_cell.MultiRNNCell(
        [get_rnn_cell(opts.lstm_size, opts.keep_prob)
         for _ in range(opts.num_layers)]
    )

    initial_state = cell.zero_state(opts.batch_size, tf.float32)

    lstm_outputs, final_state = tf.nn.dynamic_rnn(
        cell, lstm_inputs, initial_state=initial_state)

    # TODO initial_state = final_state

    seq_output = tf.concat(lstm_outputs, 1)
    x = tf.reshape(seq_output, [-1, opts.lstm_size])

    softmax_w = tf.Variable(tf.truncated_normal(
        [opts.lstm_size, opts.num_classes], stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(opts.num_classes))

    logits = tf.matmul(x, softmax_w) + softmax_b
    prediction = tf.nn.softmax(logits)

    y_one_hot = tf.one_hot(labels, opts.num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                      opts.grad_clip)
    optimizer = tf.train.AdamOptimizer(opts.learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))


def get_rnn_cell(size, keep_prob):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(size)
    lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return lstm
