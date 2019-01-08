#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def model_fn(features, labels, mode, params):
    """Build model graph."""

    opts = params['opts']

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    inputs = features['data']
    inputs.set_shape((None, opts.max_length))

    embeddings = load_word_vectors(opts.word_vectors_path)
    embed_dim = embeddings.shape[1]
    embeddings_static = tf.convert_to_tensor(
        embeddings, name='embeddings_static')
    embeddings_dynamic = tf.get_variable(
        'embeddings_dynamic', initializer=embeddings, trainable=True)

    embed_static = tf.nn.embedding_lookup(embeddings_static, inputs)

    # embed_dynamic = tf.nn.embedding_lookup(embeddings_dynamic, inputs)
    embed_dynamic = mask_padding_embedding_lookup(
        embeddings_dynamic, embed_dim, inputs)

    # (None, max_length, embed_dim, 2)
    embeds = tf.stack((embed_static, embed_dynamic), -1)

    poolings = []
    for width in opts.filter_sizes:
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
                               1-opts.dropout_keep_prob,
                               training=is_training)
    logits = tf.layers.dense(output, params['num_classes'])
    labels = tf.reshape(labels, [-1])
    labels = tf.one_hot(labels, params['num_classes'])
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('batch_accuracy', tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1)),
                    tf.float32)))

        loss = tf.losses.softmax_cross_entropy(labels, logits)
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


def load_word_vectors(filename):
    with open(filename) as f:
        tokens = f.readline().strip().split(' ')
        cnt, dim = int(tokens[0]), int(tokens[1])
        cnt += 1  # add one padding
        data = np.zeros([cnt, dim], dtype=np.float32)
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                break
            tokens = line.split(' ')
            features = map(float, tokens[1:])
            data[index+1, :] = features
        assert index == cnt - 2
    tf.logging.info("word vectors shape = {}".format(data.shape))

    return data


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
