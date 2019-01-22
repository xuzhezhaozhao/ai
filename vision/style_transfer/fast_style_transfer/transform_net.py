#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def transform_net(name, x):
    with tf.variable_scope(name):
        tf.logging.info("transform_net input {}".format(x))
        y = conv_block(x, 32, 9, 1, 'conv1')
        y = conv_block(y, 64, 3, 2, 'conv2')
        y = conv_block(y, 128, 3, 2, 'conv3')
        y = residual_block(y, 128, 'res1')
        y = residual_block(y, 128, 'res2')
        y = residual_block(y, 128, 'res3')
        y = residual_block(y, 128, 'res4')
        y = residual_block(y, 128, 'res5')
        y = upsample_conv_block(y, 64, 3, 1, 'deconv1')
        y = upsample_conv_block(y, 32, 3, 1, 'deconv2')
        y = conv_layer(y, 3, 9, 1, 'deconv3')
        y = tf.nn.tanh(y) * 150.0 + 255.0 / 2.0
        tf.logging.info("transform_net output {}".format(y))
        return y


def conv_layer(x, filters, kernel_size, stride, name='conv'):
    with tf.variable_scope(name):
        out = tf.layers.conv2d(
            x, filters, kernel_size, stride, 'SAME',
            kernel_initializer=tf.initializers.truncated_normal(0, 0.01))
        return out


def relu(x):
    return tf.nn.relu(x)


def instance_norm(x, name='in'):
    with tf.variable_scope(name):
        var_shape = [x.get_shape()[3].value]
        mu, sigma_sq = tf.nn.moments(x, [1, 2], keep_dims=True)
        shift = tf.get_variable('shift', initializer=tf.zeros(var_shape))
        scale = tf.get_variable('scale', initializer=tf.ones(var_shape))
        epsilon = 1e-3
        normalized = (x - mu) / (sigma_sq + epsilon)**(.5)
        return scale * normalized + shift


def conv_block(x, filters, kernel_size, stride, name):
    with tf.variable_scope(name):
        y = relu(instance_norm(conv_layer(x, filters, kernel_size, stride)))
        return y


def residual_block(x, filters, name='res'):
    residual = x
    with tf.variable_scope(name):
        out = instance_norm(conv_layer(x, filters, 3, 1, 'conv1'), 'in1')
        out = relu(out)
        out = instance_norm(conv_layer(out, filters, 3, 1, 'conv2'), 'in2')
        out = out + residual
        return out


def transpose_conv_layer(x, filters, kernel_size, stride, name='trans_conv'):
    with tf.variable_scope(name):
        net = tf.layers.conv2d_transpose(
            x, filters, kernel_size, stride, padding='SAME',
            kernel_initializer=tf.initializers.truncated_normal(0, 0.1))
        return net


def upsample_conv_block(x, filters, kernel_size, stride, name='deconv'):
    with tf.variable_scope(name):
        height = x.get_shape()[1].value
        width = x.get_shape()[2].value
        new_height = height * stride * 2
        new_width = width * stride * 2
        x_resized = tf.image.resize_images(
            x, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        y = conv_layer(x_resized, filters, kernel_size, stride)
        y = instance_norm(relu(y))
        return y
