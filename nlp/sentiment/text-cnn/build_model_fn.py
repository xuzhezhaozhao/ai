#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def model_fn(features, labels, mode, params):
    """Build model graph."""

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    inputs = features['data']


def load_word_vectors(filename):
    with open(filename) as f:
        tokens = f.readline().strip().split(' ')
        cnt, dim = int(tokens[0]), int(tokens[1])
        data = np.zeros([cnt, dim], dtype=np.float32)
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                break
            tokens = line.split(' ')
            features = map(float, tokens[1:])
            data[index, :] = features

        assert index == cnt-1
    print("word vectors shape = {}, dtype".format(data.shape, data.dtype))
    return data
