#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.preprocessing import normalize

import tensorflow as tf
import numpy as np

tf.app.flags.DEFINE_string('image_features', 'output.txt', '')
tf.app.flags.DEFINE_string('dict', 'input.txt', '')
tf.app.flags.DEFINE_string('output', 'nn.txt', '')
tf.app.flags.DEFINE_integer('k', 5, '')

FLAGS = tf.app.flags.FLAGS


def load_features(filename):
    data = np.loadtxt(FLAGS.image_features)
    data = normalize(data, axis=1)
    return data


def main(_):
    data = load_features(FLAGS.image_features)
    dist = np.dot(data, data.transpose())

    sorted_indices = np.flip(np.argsort(dist), axis=-1)
    sorted_indices = sorted_indices[:, 1:]

    # top-k
    nn_indices = sorted_indices[:, :FLAGS.k]  # [N, k]

    keys = [key.strip() for key in open(FLAGS.dict) if key.strip() != '']
    keys = np.array(keys)
    nn_keys = keys[nn_indices]  # [N, k]

    with open(FLAGS.output, 'w') as f:
        for i in range(len(nn_indices)):
            f.write("{}\n".format(keys[i]))
            for j in range(len(nn_indices[i, :])):
                score = dist[i, nn_indices[i, j]]
                key = nn_keys[i, j]
                f.write("{}: {}\n".format(key, score))
            f.write("\n")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run()
