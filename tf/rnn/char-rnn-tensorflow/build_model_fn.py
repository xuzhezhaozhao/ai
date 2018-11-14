#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def model_fn(features, labels, mode, params):
    """Build char rnn model graph."""

    opts = params['opts']
    vocab_size = params['vocab_size']

    # use embedding for Chinese, not necessary for English
    if not opts.use_embedding:
        lstm_inputs = tf.one_hot(features, vocab_size)
    else:
        embedding = tf.get_variable(
            'embedding', [vocab_size, opts.embedding_dim])
        lstm_inputs = tf.nn.embedding_lookup(embedding, features)

    cell = tf.nn.rnn_cell.MultiRNNCell(
        [get_rnn_cell(opts.hidden_size, opts.keep_prob)
         for _ in range(opts.num_layers)]
    )

    initial_state = cell.zero_state(opts.batch_size, tf.float32)

    # lstm_outputs: shape [batch, seq_length, hidden_size]
    # final_state: a tuple of num_layers elements of shape [batch, hidden_size]
    lstm_outputs, final_state = tf.nn.dynamic_rnn(
        cell, lstm_inputs, initial_state=initial_state)

    # add denpendecy
    # assign_op = tf.assign(initial_state, final_state)

    # shape [batch, seq_length, vocab_size]
    logits = tf.layers.dense(lstm_outputs, vocab_size)
    # predictions = tf.nn.softmax(logits)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    optimizer = tf.train.AdamOptimizer(opts.learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    if opts.use_clip_gradients:
        gradients, _ = tf.clip_by_global_norm(gradients, opts.clip_norm)
    train_op = optimizer.apply_gradients(
        zip(gradients, variables),
        global_step=tf.train.get_global_step())
    for var, grad in zip(variables, gradients):
        tf.summary.histogram(
            var.name.replace(':', '_') + '/gradient', grad)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def get_rnn_cell(size, keep_prob):
    lstm = tf.nn.rnn_cell.LSTMCell(size)
    lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return lstm
