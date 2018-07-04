#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


sess = tf.Session()

x_col = tf.feature_column.numeric_column(
    key='x', shape=[2, 1], dtype=tf.int32,
    normalizer_fn=lambda x: tf.concat([x, x], axis=1))
features = {'x': tf.constant([[1], [2], [3], [4]])}
x = tf.feature_column.input_layer(features, feature_columns=x_col)

print(sess.run(x))
