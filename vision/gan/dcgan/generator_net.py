#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import deconv, batch_norm, relu


def generator_net(x, training, opts, name='Generator'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        sz = opts.img_size // 16
        # x is input. 1 x 1 x nz
        # state size. sz x sz x (ngf*8)
        y = deconv(x, opts.ngf*8, sz, 1, 'valid', 'deconv1')
        y = batch_norm(y, training)
        y = relu(y)

        # state size. (sz*2) x (sz*2) x (ngf*4)
        y = deconv(y, opts.ngf*4, 4, 2, 'same', 'deconv2')
        y = batch_norm(y, training)
        y = relu(y)

        # state size. (sz*4) x (sz*4) x (ngf*2)
        y = deconv(y, opts.ngf*2, 4, 2, 'same', 'deconv3')
        y = batch_norm(y, training)
        y = relu(y)

        # state size. (sz*8) x (sz*8) x ngf
        y = deconv(y, opts.ngf, 4, 2, 'same', 'deconv4')
        y = batch_norm(y, training)
        y = relu(y)

        # state size. (sz*16) x (sz*16) x nc
        y = deconv(y, opts.nc, 4, 2, 'same', 'deconv5')
        y = tf.nn.tanh(y)
        tf.logging.info("Generator output: {}".format(y))

        return y
