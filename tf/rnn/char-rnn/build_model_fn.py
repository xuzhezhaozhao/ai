#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def model_fn(features, labels, mode, params):
    """Build char rnn model graph."""

    tf.summary.histogram('features', features)

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
    # final_state: a tuple of num_layers elements,
    # each shape [batch, hidden_size]
    lstm_outputs, final_state = tf.nn.dynamic_rnn(
        cell, lstm_inputs, initial_state=initial_state)

    # add denpendecy
    # assign_op = tf.assign(initial_state, final_state)

    # shape [batch, seq_length, vocab_size]
    logits = tf.layers.dense(lstm_outputs, vocab_size)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return create_predict_estimator_spec(mode, logits, labels, params)

    if mode == tf.estimator.ModeKeys.EVAL:
        return create_eval_estimator_spec(mode, logits, labels, params)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return create_train_estimator_spec(mode, logits, labels, params)


def get_rnn_cell(size, keep_prob):
    lstm = tf.nn.rnn_cell.LSTMCell(size)
    lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return lstm


def get_loss(logits, labels):

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    return loss


def create_train_estimator_spec(mode, logits, labels, params):

    opts = params['opts']
    num_samples_per_epoch = params['num_samples_per_epoch']

    loss = get_loss(logits, labels)

    global_step = tf.train.get_global_step()
    learning_rate = configure_learning_rate(
        num_samples_per_epoch, global_step, opts)
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = configure_optimizer(learning_rate, opts)

    gradients, variables = zip(*optimizer.compute_gradients(loss))
    if opts.use_clip_gradients:
        gradients, _ = tf.clip_by_global_norm(gradients, opts.clip_norm)
    train_op = optimizer.apply_gradients(
        zip(gradients, variables),
        global_step=global_step)
    for var, grad in zip(variables, gradients):
        tf.summary.histogram(
            var.name.replace(':', '_') + '/gradient', grad)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def create_eval_estimator_spec(mode, logits, labels, params):

    loss = get_loss(logits, labels)
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=None)


def create_predict_estimator_spec(mode, logits, labels, params):
    pass


def configure_optimizer(learning_rate, opts):
    """Configures the optimizer used for training."""

    if opts.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=opts.adadelta_rho,
            epsilon=opts.opt_epsilon)
    elif opts.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=opts.adagrad_initial_accumulator_value)
    elif opts.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=opts.adam_beta1,
            beta2=opts.adam_beta2,
            epsilon=opts.opt_epsilon)
    elif opts.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=opts.ftrl_learning_rate_power,
            initial_accumulator_value=opts.ftrl_initial_accumulator_value,
            l1_regularization_strength=opts.ftrl_l1,
            l2_regularization_strength=opts.ftrl_l2)
    elif opts.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=opts.momentum,
            name='Momentum')
    elif opts.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=opts.rmsprop_decay,
            momentum=opts.rmsprop_momentum,
            epsilon=opts.opt_epsilon)
    elif opts.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized' % opts.optimizer)
    return optimizer



def configure_learning_rate(num_samples_per_epoch, global_step, opts):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
    decay_steps = int(num_samples_per_epoch * opts.num_epochs_per_decay /
                      opts.batch_size)
    tf.logging.info('decay_steps = {}'.format(decay_steps))

    if opts.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(
            opts.learning_rate,
            global_step,
            decay_steps,
            opts.learning_rate_decay_factor,
            staircase=True,
            name='exponential_decay_learning_rate')
    elif opts.learning_rate_decay_type == 'fixed':
        return tf.constant(opts.learning_rate, name='fixed_learning_rate')
    elif opts.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(
            opts.learning_rate,
            global_step,
            decay_steps,
            opts.end_learning_rate,
            power=1.0,
            cycle=False,
            name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                         opts.learning_rate_decay_type)
