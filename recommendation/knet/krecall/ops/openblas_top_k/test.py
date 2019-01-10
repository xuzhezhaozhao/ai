#! /usr/bin/env python
# -*- coding=utf8 -*-

import tensorflow as tf
import numpy as np
import struct

openblas_top_k_ops = tf.load_op_library('openblas_top_k_ops.so')
openblas_top_k = openblas_top_k_ops.openblas_top_k

WEIGHTS_PATH = 'weights.bin'
BIASES_PATH = 'biases.bin'

weights = np.arange(100).reshape([20, 5]).astype(np.float)
biases = np.array([0.1]*20)


def save_numpy_float_array(array, filename):
    with open(filename, 'wb') as f:
        for d in array.shape:
            f.write(struct.pack('<q', d))

        fl = array.flat
        for v in fl:
            f.write(struct.pack('<f', v))


save_numpy_float_array(weights, WEIGHTS_PATH)
save_numpy_float_array(biases, BIASES_PATH)


sess = tf.Session()
user_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
values, indices = openblas_top_k(input=user_vector, k=5,
                                 weights_path=WEIGHTS_PATH,
                                 biases_path=BIASES_PATH)

values = sess.run(values)
indices = sess.run(indices)

print(values)
print(indices)
