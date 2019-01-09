#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def model_fn(features, labels, mode, params):
    """Build model graph."""

    opts = params['opts']

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    inputs = features['data']
    inputs.set_shape((None, opts.max_length))

    chars = [word.strip() for word in open(opts.char_dict_path)
             if word.strip() != '']
    chars.insert(0, '')

    with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
        init_width = 1.0 / opts.embedding_dim
        embeddings = tf.get_variable(
            "embeddings", initializer=tf.random_uniform(
                [len(chars), opts.embedding_dim], -init_width, init_width))
    embed_dim = embeddings.shape[1]

    embeds = mask_padding_embedding_lookup(embeddings, embed_dim, inputs)
    # (None, max_length, embed_dim, 1)
    embeds = tf.expand_dims(embeds, -1)

    poolings = []
    for width in map(int, opts.filter_sizes):
        conv = tf.layers.conv2d(embeds,
                                filters=opts.num_filters,
                                kernel_size=(width, embed_dim),
                                strides=(1, 1),
                                padding='valid')
        pool_size = (conv.shape[1].value, 1)
        pooling = tf.layers.max_pooling2d(conv,
                                          pool_size=pool_size,
                                          strides=1)
        pooling = tf.reshape(pooling, (-1, opts.num_filters))
        poolings.append(pooling)

    output = tf.concat(poolings, -1)
    output = tf.layers.dropout(output,
                               1 - opts.dropout_keep_prob,
                               training=is_training)

    logits = tf.layers.dense(output, params['num_classes'])

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'scores': tf.nn.softmax(logits),
        }
        export_outputs = {
            'predicts': tf.estimator.export.PredictOutput(outputs=predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions,
                                          export_outputs=export_outputs)

    labels = tf.reshape(labels, [-1])
    labels = tf.one_hot(labels, params['num_classes'])
    label_weights = features['label_weights']
    label_weights = tf.reshape(label_weights, [-1])

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('batch_accuracy', tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1)),
                    tf.float32)))

        loss = tf.losses.softmax_cross_entropy(
            labels, logits, weights=label_weights)
        global_step = tf.train.get_global_step()

        learning_rate = configure_learning_rate(global_step, opts)
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = configure_optimizer(learning_rate, opts)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.contrib.training.create_train_op(
            total_loss=loss,
            optimizer=optimizer,
            global_step=global_step,
            update_ops=update_ops,
            variables_to_train=tf.trainable_variables(),
            transform_grads_fn=None,
            summarize_gradients=True,
            aggregation_method=None,
            colocate_gradients_with_ops=False,
            check_numerics=True)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(
            labels=tf.argmax(labels, 1),
            predictions=tf.argmax(logits, 1))
        metrics = {
            'accuracy': accuracy
        }
        for key in metrics.keys():
            tf.summary.scalar(key, metrics[key][1])

        loss = tf.losses.softmax_cross_entropy(
            labels, logits, weights=label_weights)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)
    else:
        raise ValueError("Unsupported mode.")


def configure_learning_rate(global_step, opts):
    """Configures the learning rate.

    Args:
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
    decay_steps = opts.decay_steps

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


def mask_padding_embedding_lookup(embeddings, embedding_dim, input):
    """ mask padding tf.nn.embedding_lookup.

    ref(@ay27): https://github.com/tensorflow/tensorflow/issues/2373
    """

    mask_padding_zero_op = tf.scatter_update(
        embeddings, 0, tf.zeros([embedding_dim], dtype=tf.float32),
        name="mask_padding_zero_op")
    with tf.control_dependencies([mask_padding_zero_op]):
        output = tf.nn.embedding_lookup(
            embeddings, tf.cast(input, tf.int32, name="lookup_idx_cast"),
            name="embedding_lookup")
    return output
