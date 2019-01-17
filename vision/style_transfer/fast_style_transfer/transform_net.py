#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def transform_net(name, inputs, training):
    with tf.variable_scope(name):
        tf.logging.info("transform_net input {}".format(inputs))
        conv1 = conv('conv1', inputs, 32, 9, 1, training)
        conv2 = conv('conv2', conv1, 64, 3, 2, training)
        conv3 = conv('conv3', conv2, 128, 3, 2, training)
        res4 = res('res4', conv3, 128, 3, 1, training)
        res5 = res('res5', res4, 128, 3, 1, training)
        res6 = res('res6', res5, 128, 3, 1, training)
        res7 = res('res7', res6, 128, 3, 1, training)
        res8 = res('res8', res7, 128, 3, 1, training)
        conv_t9 = resize_conv('conv_trans9', res8, 64, 3, 2, training)
        conv_t10 = resize_conv('conv_trans10', conv_t9, 32, 3, 2, training)
        conv11 = conv('conv11', conv_t10, 3, 9, 1, training, None)
        output = tf.nn.tanh(conv11) * 150.0 + 255.0 / 2.0
        tf.logging.info("transform_net output {}".format(output))
        return output


def conv(name, inputs, filters, kernel_size, stride, training,
         activation_fn=tf.nn.relu):
    with tf.variable_scope(name):
        net = tf.layers.conv2d(
            inputs, filters, kernel_size, stride, 'SAME',
            kernel_initializer=tf.initializers.truncated_normal(0, 0.1))
        net = instance_normalization(net, training=training)
        if activation_fn:
            net = activation_fn(net)
        return net


def batch_normalization(inputs, training):
    return tf.layers.batch_normalization(inputs, training=training)


def instance_normalization(inputs, training=True):
    with tf.variable_scope('instance_normalization'):
        var_shape = [inputs.get_shape()[3].value]
        mu, sigma_sq = tf.nn.moments(inputs, [1, 2], keep_dims=True)
        shift = tf.get_variable('shift', initializer=tf.zeros(var_shape))
        scale = tf.get_variable('scale', initializer=tf.ones(var_shape))
        epsilon = 1e-3
        normalized = (inputs - mu) / (sigma_sq + epsilon)**(.5)
        return scale * normalized + shift


def res(name, inputs, filters, kernel_size, stride, training,
        activation_fn=tf.nn.relu):

    with tf.variable_scope(name):
        net = conv('conv1', inputs, filters, kernel_size, stride, training)
        net = conv('conv2', net, filters, kernel_size, stride, training,
                   activation_fn=None)
        net = inputs + net
        return net


def conv_transpose(name, inputs, filters, kernel_size, stride, training,
                   activation_fn=tf.nn.relu):
    with tf.variable_scope(name):
        net = tf.layers.conv2d_transpose(
            inputs, filters, kernel_size, stride, padding='SAME',
            kernel_initializer=tf.initializers.truncated_normal(0, 0.1))
        net = instance_normalization(net, training=training)
        if activation_fn:
            net = activation_fn(net)
        return net


def resize_conv(name, x, filters, kernel_size, stride, training,
                activation_fn=tf.nn.relu):
    with tf.variable_scope(name):
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        new_height = height * stride * 2
        new_width = width * stride * 2

        x_resized = tf.image.resize_images(
            x, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return conv('conv', x_resized, filters, kernel_size, stride, training,
                    activation_fn)
