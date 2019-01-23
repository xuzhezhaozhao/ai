#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import conv, leaky_relu, batch_norm


def discriminator_net(x, training, opts, name='Discriminator'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # input is 64 x 64 x nc
        # state size. 32 x 32 x ndf
        y = conv(x, opts.ndf, 4, 2, 'same', 'conv1')
        y = leaky_relu(y, 0.2)

        # state size. 16 x 16 x (ndf*2)
        y = conv(y, opts.ndf*2, 4, 2, 'same', 'conv2')
        y = batch_norm(y, training)
        y = leaky_relu(y, 0.2)

        # state size. 8 x 8 x (ndf*4)
        y = conv(y, opts.ndf*4, 4, 2, 'same', 'conv3')
        y = batch_norm(y, training)
        y = leaky_relu(y, 0.2)

        # state size. 4 x 4 x (ndf*8)
        y = conv(y, opts.ndf*8, 4, 2, 'same', 'conv4')
        y = batch_norm(y, training)
        y = leaky_relu(y, 0.2)

        # output
        y = conv(y, 1, 4, 1, 'valid', 'conv5')
        logits = tf.reshape(y, (-1, 1))

        return logits
