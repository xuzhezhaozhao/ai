#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def model_fn(features, labels, mode, params):
    """Build model graph."""

    opts = params['opts']

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    inputs = features['data']
    inputs.set_shape([None, opts.max_length])

    embeddings = load_word_vectors(opts.word_vectors_path)
    embed_dim = embeddings.shape[1]
    embeddings_static = tf.convert_to_tensor(
        embeddings, name='embeddings_static')
    embeddings_dynamic = tf.get_variable(
        'embeddings_dynamic', initializer=embeddings, trainable=True)

    embed_static = tf.nn.embedding_lookup(embeddings_static, inputs)
    embed_dynamic = tf.nn.embedding_lookup(embeddings_dynamic, inputs)
    embeds = tf.stack([embed_static, embed_dynamic], -1)

    outputs = tf.layers.conv2d(embeds,
                               filters=1,
                               kernel_size=[2, embed_dim],
                               strides=[1, 1],
                               padding='valid')
    pool_size = (outputs.shape[1].value, 1)
    outputs = tf.layers.max_pooling2d(outputs, pool_size=pool_size, strides=1)
    outputs = tf.squeeze(outputs)
    print(outputs)


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

        assert index == cnt - 1
    print("word vectors shape = {}, dtype".format(data.shape, data.dtype))
    return data
