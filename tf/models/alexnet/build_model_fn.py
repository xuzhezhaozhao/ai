#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def alexnet_model_fn(features, labels, mode, params):
    """Build model graph."""

    opts = params['opts']
    weights_dict = load_initial_weights(opts)

    # Create the network graph.
    # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
    pre_weights = weights_dict['conv1'][0]
    pre_biases = weights_dict['conv1'][1]
    trainable = True if 'conv1' in opts.train_layers else False
    conv1 = conv(features, 11, 11, 96, 4, 4, padding='VALID', name='conv1',
                 trainable=trainable,
                 pre_weights=pre_weights, pre_biases=pre_biases)
    norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
    pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

    # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
    pre_weights = weights_dict['conv2'][0]
    pre_biases = weights_dict['conv2'][1]
    trainable = True if 'conv2' in opts.train_layers else False
    conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2',
                 trainable=trainable,
                 pre_weights=pre_weights, pre_biases=pre_biases)
    norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
    pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

    # 3rd Layer: Conv (w ReLu)
    pre_weights = weights_dict['conv3'][0]
    pre_biases = weights_dict['conv3'][1]
    trainable = True if 'conv3' in opts.train_layers else False
    conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3',
                 trainable=trainable,
                 pre_weights=pre_weights, pre_biases=pre_biases)

    # 4th Layer: Conv (w ReLu) splitted into two groups
    pre_weights = weights_dict['conv4'][0]
    pre_biases = weights_dict['conv4'][1]
    trainable = True if 'conv4' in opts.train_layers else False
    conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4',
                 trainable=trainable,
                 pre_weights=pre_weights, pre_biases=pre_biases)

    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    pre_weights = weights_dict['conv5'][0]
    pre_biases = weights_dict['conv5'][1]
    trainable = True if 'conv5' in opts.train_layers else False
    conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5',
                 trainable=trainable,
                 pre_weights=pre_weights, pre_biases=pre_biases)
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
    pre_weights = weights_dict['fc6'][0]
    pre_biases = weights_dict['fc6'][1]
    trainable = True if 'fc6' in opts.train_layers else False
    fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6',
             trainable=trainable,
             pre_weights=pre_weights, pre_biases=pre_biases)

    dropout6 = dropout(fc6, 1-opts.dropout)

    # 7th Layer: FC (w ReLu) -> Dropout
    pre_weights = weights_dict['fc7'][0]
    pre_biases = weights_dict['fc7'][1]
    trainable = True if 'fc7' in opts.train_layers else False
    fc7 = fc(dropout6, 4096, 4096, name='fc7',
             trainable=trainable,
             pre_weights=pre_weights, pre_biases=pre_biases)
    dropout7 = dropout(fc7, 1-opts.dropout)

    # 8th Layer: FC and return unscaled activations
    pre_weights = weights_dict['fc8'][0]
    pre_biases = weights_dict['fc8'][1]
    trainable = True if 'fc8' in opts.train_layers else False
    fc8 = fc(dropout7, 4096, opts.num_classes, relu=False, name='fc8',
             trainable=trainable,
             pre_weights=pre_weights, pre_biases=pre_biases)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return create_predict_estimator_spec(mode)

    if mode == tf.estimator.ModeKeys.EVAL:
        return create_eval_estimator_spec(mode, fc8, labels, params)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return create_train_estimator_spec(mode, fc8, labels, params)


def load_initial_weights(opts):
    """Load weights from file.

    As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
    come as a dict of lists (e.g. weights['conv1'] is a list) and not as
    dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
    'biases') we need a special load function
    """

    weights_dict = np.load(opts.pretrained_weights_path,
                           encoding='bytes').item()
    return weights_dict


def conv(x, filter_height, filter_width, num_filters,
         stride_y, stride_x, name,
         padding='SAME', groups=1,
         trainable=True, pre_weights=None, pre_biases=None):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """

    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    def convolve(i, k): return tf.nn.conv2d(
            i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    if trainable:
        weights_shape = [filter_height, filter_width,
                         input_channels/groups, num_filters]
        biases_shape = [num_filters]
        pre_weights = None
        pre_biases = None
    else:
        weights_shape = None
        biases_shape = None

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights',
                                  initializer=pre_weights,
                                  shape=weights_shape,
                                  trainable=trainable)
        biases = tf.get_variable('biases',
                                 initializer=pre_biases,
                                 shape=biases_shape,
                                 trainable=trainable)

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
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
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, name, relu=True,
       trainable=True, pre_weights=None, pre_biases=None):
    """Create a fully connected layer."""

    if trainable:
        weights_shape = [num_in, num_out]
        biases_shape = [num_out]
        pre_weights = None
        pre_biases = None
    else:
        weights_shape = None
        biases_shape = None

    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights',
                                  initializer=pre_weights,
                                  shape=weights_shape,
                                  trainable=trainable)
        biases = tf.get_variable('biases',
                                 initializer=pre_biases,
                                 shape=biases_shape,
                                 trainable=trainable)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""

    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""

    return tf.nn.local_response_normalization(
        x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""

    return tf.nn.dropout(x, keep_prob)


def create_predict_estimator_spec(mode):
    pass


def create_eval_estimator_spec(mode, score, labels, params):
    """Create eval EstimatorSpec."""
    accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                   predictions=tf.argmax(score, 1))
    metrics = {
        'accuracy': accuracy
    }

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=labels))

    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


def create_train_estimator_spec(mode, score, labels, params):
    """Create train EstimatorSpec."""

    opts = params['opts']

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=labels))
    optimizer = tf.train.GradientDescentOptimizer(opts.lr)

    gradients, variables = zip(*optimizer.compute_gradients(
        loss, gate_gradients=tf.train.Optimizer.GATE_GRAPH))
    train_op = optimizer.apply_gradients(
        zip(gradients, variables), global_step=tf.train.get_global_step())

    for var, grad in zip(variables, gradients):
        tf.summary.histogram(var.name.replace(':', '_') + '/gradient', grad)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
