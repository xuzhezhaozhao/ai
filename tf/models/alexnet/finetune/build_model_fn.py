#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import model_keys


def alexnet_model_fn(features, labels, mode, params):
    """Build model graph."""

    tf.summary.image('images', features[model_keys.DATA_COL])

    opts = params['opts']
    weights_dict = load_pretrained_weights(opts)
    data = features[model_keys.DATA_COL]

    # Create the network graph.
    # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
    conv1 = conv_layer(weights_dict, opts, data, 11, 11, 96, 4, 4,
                       padding='VALID', name='conv1', groups=1)
    norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
    pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

    # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
    conv2 = conv_layer(weights_dict, opts, pool1, 5, 5, 256, 1, 1,
                       padding='SAME', name='conv2', groups=2)
    norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
    pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

    # 3rd Layer: Conv (w ReLu)
    conv3 = conv_layer(weights_dict, opts, pool2, 3, 3, 384, 1, 1,
                       padding='SAME', name='conv3', groups=1)

    # 4th Layer: Conv (w ReLu) splitted into two groups
    conv4 = conv_layer(weights_dict, opts, conv3, 3, 3, 384, 1, 1,
                       padding='SAME', name='conv4', groups=2)

    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    conv5 = conv_layer(weights_dict, opts, conv4, 3, 3, 256, 1, 1,
                       padding='SAME', name='conv5', groups=2)
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

    training = (mode == tf.estimator.ModeKeys.TRAIN)

    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    trainable = True if 'fc6' in opts.train_layers else False
    fc6 = fc_layer(weights_dict, opts, pool5, name='fc6', True, training)
    dropout6 = dropout(fc6, opts.dropout, training) if trainable else fc6

    # 7th Layer: FC (w ReLu) -> Dropout
    trainable = True if 'fc7' in opts.train_layers else False
    fc7 = fc_layer(weights_dict, opts, dropout6, name='fc7', True, training)
    dropout7 = dropout(fc7, opts.dropout, training) if trainable else fc7

    # 8th Layer: FC and return unscaled activations
    trainable = True if 'fc8' in opts.train_layers else False
    fc8 = fc_layer(weights_dict, opts, dropout7, name='fc8', False, training)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return create_predict_estimator_spec(mode, fc8, labels, params)

    if mode == tf.estimator.ModeKeys.EVAL:
        return create_eval_estimator_spec(mode, fc8, labels, params)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return create_train_estimator_spec(mode, fc8, labels, params)


def load_pretrained_weights(opts):
    """Load weights from file.

    As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
    come as a dict of lists (e.g. weights['conv1'] is a list) and not as
    dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
    'biases') we need a special load function
    """

    weights_dict = np.load(opts.pretrained_weights_path,
                           encoding='bytes').item()
    return weights_dict


def conv_layer(weights_dict, opts, x, filter_height, filter_width, num_filters,
               stride_y, stride_x, name, padding, groups):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """

    # Create lambda function for the convolution
    def convolve(i, k): return tf.nn.conv2d(
            i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name):
        weights = get_conv_filter(name, weights_dict, opts)
        biases = get_bias(name, weights_dict, opts)

    if groups == 1:
        conv = convolve(x, weights)
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k)
                         for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias)

    return relu


def fc_layer(weights_dict, opts, x, name, relu, training):
    """Create a fully connected layer."""

    with tf.variable_scope(name):
        shape = x.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(x, [-1, dim])

        weights = get_fc_weight(name, weights_dict, opts)
        biases = get_bias(name, weights_dict, opts)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding):
    """Create a max pooling layer."""

    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""

    return tf.nn.local_response_normalization(
        x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)


def dropout(x, rate, training):
    """Create a dropout layer."""

    return tf.layers.dropout(x, rate, training=training)


def cross_entropy(score, labels):
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=score, labels=tf.argmax(labels, 1)))
    return loss


def create_predict_estimator_spec(mode, score, labels, params):
    """Create predict EstimatorSpec."""

    score = tf.nn.softmax(score)
    predictions = {
        'score': score,
    }

    export_outputs = {
        'predicts': tf.estimator.export.PredictOutput(
            outputs={
                'score': score,
            }
        )
    }

    return tf.estimator.EstimatorSpec(mode, predictions=predictions,
                                      export_outputs=export_outputs)


def create_eval_estimator_spec(mode, score, labels, params):
    """Create eval EstimatorSpec."""

    accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                   predictions=tf.argmax(score, 1))
    metrics = {
        'accuracy': accuracy
    }

    loss = cross_entropy(score, labels)

    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


def create_train_estimator_spec(mode, score, labels, params):
    """Create train EstimatorSpec."""

    tf.summary.scalar('train_accuracy', tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(score, 1)),
                tf.float32)))

    opts = params['opts']
    global_step = tf.train.get_global_step()
    loss = cross_entropy(score, labels)
    lr = tf.train.exponential_decay(
        opts.lr,
        global_step,
        decay_steps=opts.optimizer_exponential_decay_steps,
        decay_rate=opts.optimizer_exponential_decay_rate,
        staircase=opts.optimizer_exponential_decay_staircase)
    tf.summary.scalar('lr', lr)
    optimizer = tf.train.MomentumOptimizer(
        lr, opts.optimizer_momentum_momentum)
    gradients, variables = zip(*optimizer.compute_gradients(
        loss, gate_gradients=tf.train.Optimizer.GATE_GRAPH))
    train_op = optimizer.apply_gradients(
        zip(gradients, variables), global_step=global_step)

    for var, grad in zip(variables, gradients):
        tf.summary.histogram(var.name.replace(':', '_') + '/gradient', grad)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def get_conv_filter(name, weights_dict, opts):
    trainable = True if name in opts.train_layers else False
    return tf.get_variable(
        'filter', initializer=weights_dict[name][0], trainable=trainable)


def get_bias(name, weights_dict, opts):
    trainable = True if name in opts.train_layers else False
    if not trainable or name != 'fc8':
        biases = tf.get_variable(
            'biases', initializer=weights_dict[name][1], trainable=trainable)
    else:
        biases = tf.get_variable(
            'biases', shape=[opts.num_classes], trainable=trainable)

    return biases


def get_fc_weight(name, weights_dict, opts):
    trainable = True if name in opts.train_layers else False
    if not trainable or name != 'fc8':
        weights = tf.get_variable(
            'weights', initializer=weights_dict[name][0], trainable=trainable)
    else:  # trainable and name == 'fc8'
        weights = tf.get_variable(
            'weights',
            shape=[weights_dict[name][0].shape[0], opts.num_classes],
            trainable=trainable)

    return weights
