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

    batch_size = opts.batch_size
    if mode == tf.estimator.ModeKeys.PREDICT:
        batch_size = 1

    states = get_state_variables(batch_size, cell)

    # lstm_outputs: shape [batch, seq_length, hidden_size]
    # new_state: a tuple of num_layers elements,
    # each shape [batch, hidden_size]
    lstm_outputs, new_states = tf.nn.dynamic_rnn(
        cell, lstm_inputs, initial_state=states)

    # Add an operation to update the train states with the last state tensors.
    update_op = get_state_update_op(states, new_states)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)

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

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
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
    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    vocab_size = params['vocab_size']
    logits = tf.reshape(logits, [-1, vocab_size])

    predictions = tf.nn.softmax(logits)
    predictions = predictions / temperature
    predicted_id = tf.multinomial(predictions, num_samples=1)
    predicted_id = predicted_id[-1, 0]
    predicted_id = tf.expand_dims(predicted_id, 0)
    predictions = {
        'predicted_id': predicted_id,
    }

    return tf.estimator.EstimatorSpec(mode, predictions=predictions)


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


# see https://stackoverflow.com/questions/37969065/tensorflow-best-way-to-save-state-in-rnns?noredirect=1&lq=1
def get_state_variables(batch_size, cell):
    # For each layer, get the initial state and make a variable out of it
    # to enable updating its value.
    state_variables = []
    for state_c, state_h in cell.zero_state(batch_size, tf.float32):
        state_variables.append(tf.contrib.rnn.LSTMStateTuple(
            tf.Variable(state_c, trainable=False),
            tf.Variable(state_h, trainable=False)))

    # Return as a tuple, so that it can be fed to dynamic_rnn as an initial
    # state.
    return tuple(state_variables)


def get_state_update_op(state_variables, new_states):
    # Add an operation to update the train states with the last state tensors.
    update_ops = []
    for state_variable, new_state in zip(state_variables, new_states):
        # Assign the new state to the state variables on this layer
        update_ops.extend([state_variable[0].assign(new_state[0]),
                           state_variable[1].assign(new_state[1])])
    return tf.group(*update_ops)


def get_state_reset_op(state_variables, cell, batch_size):
    # Return an operation to set each variable in a list of LSTMStateTuples to
    # zero.
    zero_states = cell.zero_state(batch_size, tf.float32)
    return get_state_update_op(state_variables, zero_states)
