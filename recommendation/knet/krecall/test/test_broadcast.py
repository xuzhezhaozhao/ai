#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


# 出发点是解决不共用负采样问题


sess = tf.Session()

batch_size = 2
dim = 4
num_sampled = 4

inputs = np.array(range(batch_size*dim))
inputs = inputs.reshape([batch_size, dim])


weights = np.ones([batch_size, num_sampled, dim])

weights = np.array(range(batch_size*num_sampled*dim))
weights = weights.reshape([batch_size, num_sampled, dim])

inputs = inputs.reshape([batch_size, 1, dim])  # broadcast
logits = tf.multiply(inputs, weights)
logits = tf.reduce_sum(logits, -1)

logits = sess.run(logits)
print(logits)
