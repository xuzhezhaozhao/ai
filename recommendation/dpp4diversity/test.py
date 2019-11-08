#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from dpp import dpp
from samson_dpp import dpp as sdpp


N = 10
L = np.random.random([N, N])

print("L")
print(L)


L = tf.constant(L)
K = N

dpp_indices = dpp(L, K)
sdpp_indices = sdpp(L, K)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

dpp_indices, sdpp_indices = sess.run([dpp_indices, sdpp_indices])
print("dpp")
print(dpp_indices)
print("sdpp")
print(sdpp_indices)
