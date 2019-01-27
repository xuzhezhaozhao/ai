#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import conv, leaky_relu


# No batch normalization
def discriminator_net(x, training, opts, name='Discriminator'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        sz = opts.img_size // 16
        # input is (sz*16) x (sz*16) x nc
        # state size. (sz*8) x (sz*8) x ndf
        y = conv(x, opts.ndf, 4, 2, 'same', 'conv1')
        y = leaky_relu(y, 0.2)

        # state size. (sz*4) x (sz*4) x (ndf*2)
        y = conv(y, opts.ndf * 2, 4, 2, 'same', 'conv2')
        y = leaky_relu(y, 0.2)

        # state size. (sz*2) x (sz*2) x (ndf*4)
        y = conv(y, opts.ndf * 4, 4, 2, 'same', 'conv3')
        y = leaky_relu(y, 0.2)

        # state size. sz x sz x (ndf*8)
        y = conv(y, opts.ndf * 8, 4, 2, 'same', 'conv4')
        y = leaky_relu(y, 0.2)

        flatten = tf.reshape(y, (opts.batch_size, sz * sz * opts.ndf * 8))

        # discriminator output
        logits = conv(y, 1, sz, 1, 'valid', 'conv5')
        logits = tf.reshape(y, (-1, 1))

    with tf.variable_scope('MutualInfo', reuse=tf.AUTO_REUSE):
        # Q output
        dim = opts.num_categorical + opts.num_continuous
        q = tf.layers.dense(flatten, dim)

        return logits, q
