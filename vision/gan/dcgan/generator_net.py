#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import deconv, batch_norm, relu


def generator_net(x, training, opts):
    # x is input. 1 x 1 x nz
    # state size. 4 x 4 x (ngf*8)
    y = deconv(x, opts.ngf*8, 4, 1, padding='valid')
    y = batch_norm(y, training)
    y = relu(y)

    # state size. 8 x 8 x (ngf*4)
    y = deconv(y, opts.ngf*4, 4, 2)
    y = batch_norm(y, training)
    y = relu(y)

    # state size. 16 x 16 x (ngf*2)
    y = deconv(y, opts.ngf*2, 4, 2)
    y = batch_norm(y, training)
    y = relu(y)

    # state size. 32 x 32 x ngf
    y = deconv(y, opts.ngf, 4, 2)
    y = batch_norm(y, training)
    y = relu(y)

    # state size. 64 x 64 x nc
    y = deconv(y, opts.nc, 4, 2)

    # output. 64 x 64 x nc
    y = tf.nn.tanh(y)

    return y
