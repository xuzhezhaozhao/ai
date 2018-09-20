#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

krank_input_ops = tf.load_op_library('./ops/input/krank_input_ops.so')
krank_input = krank_input_ops.krank_input
