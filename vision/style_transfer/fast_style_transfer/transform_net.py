#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def transform_net(name, inputs, training):
    tf.logging.info("transform_net input {}".format(inputs))
    with tf.variable_scope(name):
        conv1 = conv('conv1', inputs, 32, 9, 1, training)
        conv2 = conv('conv2', conv1, 64, 3, 2, training)
        conv3 = conv('conv3', conv2, 128, 3, 2, training)
        res4 = res('res4', conv3, 128, 3, 1, training)
        res5 = res('res5', res4, 128, 3, 1, training)
        res6 = res('res6', res5, 128, 3, 1, training)
        res7 = res('res7', res6, 128, 3, 1, training)
        res8 = res('res8', res7, 128, 3, 1, training)
        conv_trans9 = conv_transpose('conv_trans9', res8, 64, 3, 2, training)
        conv_trans10 = conv_transpose('conv_trans10', conv_trans9, 32, 3, 2,
                                      training)
        conv11 = conv('conv11', conv_trans10, 3, 9, 1, training, None)
        output = tf.nn.tanh(conv11) * 150.0 + 255.0 / 2.0
        tf.logging.info("transform_net output {}".format(output))
        return output


def conv(name, inputs, filters, kernel_size, stride, training,
         activation_fn=tf.nn.relu):
    with tf.variable_scope(name):
        net = tf.layers.conv2d(inputs, filters, kernel_size, stride, 'SAME')
        net = normalization(net, training=training)
        if activation_fn:
            net = activation_fn(net)
        return net


def normalization(inputs, training):
    return tf.layers.batch_normalization(inputs, training=training)


def res(name, inputs, filters, kernel_size, stride, training):
    with tf.variable_scope(name):
        conv1 = conv('conv1', inputs, filters, kernel_size, stride, training)
        conv2 = conv('conv2', conv1, filters, kernel_size, stride, training,
                     activation_fn=None)
        return conv1 + conv2


def conv_transpose(name, inputs, filters, kernel_size, stride, training,
                   activation_fn=tf.nn.relu):
    with tf.variable_scope(name):
        net = tf.layers.conv2d_transpose(inputs, filters, kernel_size, stride,
                                         padding='SAME')
        net = normalization(net, training=training)
        if activation_fn:
            net = activation_fn(net)
        return net
