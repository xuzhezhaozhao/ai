#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import model_keys


# build vgg19
def vgg19_model_fn(features, labels, mode, params):
    """Build model graph."""

    tf.summary.image('images', features[model_keys.DATA_COL])

    opts = params['opts']
    weights_dict = load_pretrained_weights(opts)
    data = features[model_keys.DATA_COL]

    assert data.get_shape().as_list()[1:] == [224, 224, 3]

    # Create the network graph.
    conv1_1 = conv_layer(data, "conv1_1", weights_dict, opts)
    conv1_2 = conv_layer(conv1_1, "conv1_2", weights_dict, opts)
    pool1 = max_pool(conv1_2, 'pool1')

    conv2_1 = conv_layer(pool1, "conv2_1", weights_dict, opts)
    conv2_2 = conv_layer(conv2_1, "conv2_2", weights_dict, opts)
    pool2 = max_pool(conv2_2, 'pool2')

    conv3_1 = conv_layer(pool2, "conv3_1", weights_dict, opts)
    conv3_2 = conv_layer(conv3_1, "conv3_2", weights_dict, opts)
    conv3_3 = conv_layer(conv3_2, "conv3_3", weights_dict, opts)
    conv3_4 = conv_layer(conv3_3, "conv3_4", weights_dict, opts)
    pool3 = max_pool(conv3_4, 'pool3')

    conv4_1 = conv_layer(pool3, "conv4_1", weights_dict, opts)
    conv4_2 = conv_layer(conv4_1, "conv4_2", weights_dict, opts)
    conv4_3 = conv_layer(conv4_2, "conv4_3", weights_dict, opts)
    conv4_4 = conv_layer(conv4_3, "conv4_4", weights_dict, opts)
    pool4 = max_pool(conv4_4, 'pool4')

    conv5_1 = conv_layer(pool4, "conv5_1", weights_dict, opts)
    conv5_2 = conv_layer(conv5_1, "conv5_2", weights_dict, opts)
    conv5_3 = conv_layer(conv5_2, "conv5_3", weights_dict, opts)
    conv5_4 = conv_layer(conv5_3, "conv5_4", weights_dict, opts)
    pool5 = max_pool(conv5_4, 'pool5')

    training = (mode == tf.estimator.ModeKeys.TRAIN)
    fc6 = fc_layer(pool5, "fc6", weights_dict, opts, training)
    relu6 = tf.nn.relu(fc6)

    fc7 = fc_layer(relu6, "fc7", weights_dict, opts, training)
    relu7 = tf.nn.relu(fc7)

    fc8 = fc_layer(relu7, "fc8", weights_dict, opts, training)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return create_predict_estimator_spec(mode, fc8, labels, params)

    if mode == tf.estimator.ModeKeys.EVAL:
        return create_eval_estimator_spec(mode, fc8, labels, params)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return create_train_estimator_spec(mode, fc8, labels, params)


def load_pretrained_weights(opts):
    """Load weights from file.
    """

    weights_dict = np.load(opts.pretrained_weights_path,
                           encoding='latin1').item()
    tf.logging.info('npy file loaded')
    return weights_dict


def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name=name)


def conv_layer(bottom, name, weights_dict, opts):
    with tf.variable_scope(name):
        filt = get_conv_filter(name, weights_dict, opts)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = get_bias(name, weights_dict, opts)
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        return relu


def fc_layer(bottom, name, weights_dict, opts, training):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])

        weights = get_fc_weight(name, weights_dict, opts)
        biases = get_bias(name, weights_dict, opts)

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc


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


def dropout(x, keep_prob):
    """Create a dropout layer."""

    return tf.nn.dropout(x, keep_prob)


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
