#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def conv(x, filters, kernel_size, stride, padding='same'):
    y = tf.layers.conv2d(x, filters, kernel_size, stride, padding)
    return y


def deconv(x, filters, kernel_size, stride, padding='same'):
    y = tf.layers.conv2d_transpose(x, filters, kernel_size, stride, padding)
    return y


def batch_norm(x, training):
    y = tf.layers.batch_normalization(x, training=training)
    return y


def relu(x):
    y = tf.nn.relu(x)
    return y


def leaky_relu(x, alpha):
    y = tf.nn.leaky_relu(x, alpha)
    return y
