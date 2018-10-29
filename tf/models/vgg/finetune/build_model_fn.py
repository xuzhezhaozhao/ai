#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import model_keys

_training_lr_decay_rate = 1.0


def set_training_lr_decay_rate(rate):
    global _training_lr_decay_rate
    _training_lr_decay_rate = rate


def get_training_lr_decay_rate():
    global _training_lr_decay_rate
    return _training_lr_decay_rate


# build vgg19
def vgg19_model_fn(features, labels, mode, params):
    """Build model graph."""

    tf.summary.image('images', features[model_keys.DATA_COL])

    opts = params['opts']
    weights_dict = load_pretrained_weights(opts)
    data = features[model_keys.DATA_COL]

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

    if (not training) and opts.multi_scale_predict:
        fc6 = fc_to_conv_layer(
            weights_dict, opts, pool5, 7, 7, 512, 4096, name='fc6')
        relu6 = tf.nn.relu(fc6)
        fc7 = fc_to_conv_layer(
            weights_dict, opts, relu6, 1, 1, 4096, 4096, name='fc7')
        relu7 = tf.nn.relu(fc7)
        fc8 = fc_to_conv_layer(
            weights_dict, opts, relu7, 1, 1, 4096,
            opts.num_classes, name='fc8')
        # fc8 shape [batch, m, n, opts.num_classes]
        fc8 = tf.reduce_mean(fc8, axis=-2)
        fc8 = tf.reduce_mean(fc8, axis=-2)
        score = tf.nn.softmax(fc8)
    else:
        fc6 = fc_layer(pool5, "fc6", weights_dict, opts, training)
        relu6 = tf.nn.relu(fc6)
        relu6 = maybe_dropout(relu6, opts.dropout, "fc6", training, opts)

        fc7 = fc_layer(relu6, "fc7", weights_dict, opts, training)
        relu7 = tf.nn.relu(fc7)
        relu7 = maybe_dropout(relu7, opts.dropout, "fc7", training, opts)

        fc8 = fc_layer(relu7, "fc8", weights_dict, opts, training)
        score = tf.nn.softmax(fc8)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return create_predict_estimator_spec(mode, score, labels, params)

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


def fc_to_conv_layer(weights_dict, opts, x, filter_height, filter_width,
                     in_channels, out_channels, name):
    """Create a fully connected layer."""

    with tf.variable_scope(name):
        weights = get_fc_weight(name, weights_dict, opts)
        biases = get_bias(name, weights_dict, opts)

        weights = tf.reshape(
            weights, [filter_height, filter_width, in_channels, out_channels])

        conv = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='VALID')
        bias = tf.nn.bias_add(conv, biases)

        return bias


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


def cross_entropy(logits, labels):
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.argmax(labels, 1)))
    return loss


def create_predict_estimator_spec(mode, score, labels, params):
    """Create predict EstimatorSpec."""

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


def create_eval_estimator_spec(mode, logits, labels, params):
    """Create eval EstimatorSpec."""

    accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                   predictions=tf.argmax(logits, 1))
    metrics = {
        'accuracy': accuracy
    }
    add_metrics_summary(metrics)

    loss = cross_entropy(logits, labels)
    tf.summary.scalar("eval_loss", loss)
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


def create_train_estimator_spec(mode, logits, labels, params):
    """Create train EstimatorSpec."""

    tf.summary.scalar('train_accuracy', tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1)),
                tf.float32)))

    opts = params['opts']
    global_step = tf.train.get_global_step()
    loss = cross_entropy(logits, labels)
    tf.summary.scalar("train_loss", loss)
    lr = create_global_learning_rate(opts)
    lr = tf.assign(lr, lr*get_training_lr_decay_rate())
    tf.summary.scalar('lr', lr)
    optimizer = tf.train.MomentumOptimizer(
        lr, opts.optimizer_momentum_momentum)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    train_op = optimizer.apply_gradients(
        zip(gradients, variables), global_step=global_step)

    for var, grad in zip(variables, gradients):
        tf.summary.histogram(var.name.replace(':', '_') + '/gradient', grad)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def create_global_learning_rate(opts):
    with tf.variable_scope('learning_rate'):
        lr = tf.get_variable('lr', initializer=opts.lr, trainable=False)
    return lr


def add_metrics_summary(metrics):
    """Add metrics to tensorboard."""

    for key in metrics.keys():
        tf.summary.scalar(key, metrics[key][1])


def maybe_dropout(x, rate, name, training, opts):
    trainable = True if name in opts.train_layers else False
    if trainable and rate > 0.0:
        x = tf.layers.dropout(x, rate, training=training)
    return x
