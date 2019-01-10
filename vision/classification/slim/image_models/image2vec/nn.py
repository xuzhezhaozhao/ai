#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from sklearn.preprocessing import normalize

import tensorflow as tf
import numpy as np


tf.app.flags.DEFINE_string(
    'image_features', 'features.txt', 'features file output by image2vec.')

tf.app.flags.DEFINE_string('output', 'nn.txt', 'output nn file')
tf.app.flags.DEFINE_integer('k', 10, 'top k nearest images will be ouput.')
tf.app.flags.DEFINE_string('output_format', 'human', 'human or hbcf')

tf.app.flags.DEFINE_integer('chunk_size', 128, '')

FLAGS = tf.app.flags.FLAGS


def check_data(filename):
    cnt = 0
    dim = None
    for line in open(filename):
        line = line.strip()
        if not line:
            break
        cnt += 1
        tokens = line.split(' ')
        if len(tokens) < 8:
            raise ValueError("features file format error.")
        tokens = tokens[1:]
        if dim is None:
            dim = len(tokens)
        else:
            if dim != len(tokens):
                raise ValueError("features file format error.")

    return cnt, dim


def load_features(filename):
    cnt, dim = check_data(filename)
    data = np.zeros([cnt, dim], dtype=np.float32)
    print("before load data size = {}".format(data.nbytes))
    keys = []
    for index, line in enumerate(open(filename)):
        if (index+1) % 1000 == 0:
            print("load {} lines ...".format(index+1))
        line = line.strip()
        if not line:
            break
        tokens = line.split(' ')
        keys.append(tokens[0])
        features = map(float, tokens[1:])
        data[index, :] = features

    assert index == cnt-1

    print("features shape = {}, dtype".format(data.shape, data.dtype))
    print("after load data size = {}".format(data.nbytes))
    data = normalize(data, axis=1)
    return data, keys


def chunking_dot(big_matrix, small_matrix, top_k, chunk_size=128):
    # Make a copy if the array is not already contiguous
    small_matrix = np.ascontiguousarray(small_matrix)
    N = big_matrix.shape[0]
    top_k = min(top_k, N)
    dist = np.zeros((N, top_k), dtype=np.float32)
    nn_indices = np.zeros((N, top_k), dtype=np.int64)
    cnt = 0
    for i in range(0, dist.shape[0], chunk_size):
        if cnt % 20 == 0:
            print("nn {}/{} ...".format(chunk_size*cnt, N))
        cnt += 1

        end = i + chunk_size
        sub_dist = np.dot(big_matrix[i:end], small_matrix)
        sorted_indices = np.flip(np.argsort(sub_dist), axis=-1)
        sorted_indices = sorted_indices[:, 1:]  # filter self
        sorted_indices = sorted_indices[:, :top_k]  # get top-k
        nn_indices[i:end] = sorted_indices
        for index, indice in enumerate(sorted_indices):
            dist[i+index] = sub_dist[index, indice]
    return dist, nn_indices


def main(_):
    tf.logging.info('{} Start nn ...'.format(datetime.now()))

    data, keys = load_features(FLAGS.image_features)
    keys = np.array(keys)

    # dist = np.dot(data, data.transpose())
    dist, nn_indices = chunking_dot(
        data, data.transpose(), FLAGS.k, FLAGS.chunk_size)
    nn_keys = keys[nn_indices]  # [N, k]

    with open(FLAGS.output, 'w') as f:
        for i in range(len(nn_indices)):
            if FLAGS.output_format == 'human':
                f.write("{}\n".format(keys[i]))

            for j in range(len(nn_indices[i, :])):
                score = dist[i, j]
                key = nn_keys[i, j]
                f.write("{} {}\n".format(key, score))

            if FLAGS.output_format == 'human':
                f.write("\n")

    tf.logging.info('{} Finish nn.'.format(datetime.now()))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
