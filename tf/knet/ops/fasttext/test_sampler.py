#! /usr/bin/env python
# -*- coding=utf8 -*-

import tensorflow as tf


fasttext_negative_sampler_ops = tf.load_op_library(
    'fasttext_negative_sampler_ops.so')

sess = tf.Session()

labels = [[1], [0], [2]]
unigrams = tf.make_tensor_proto([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                dtype=tf.int64)
sampled_ids = fasttext_negative_sampler_ops.fasttext_negative_sampler(
    true_classes=labels,
    num_true=1,
    num_sampled=4,
    unique=True,
    range_max=10,
    unigrams=unigrams)

sampled_ids = sess.run(sampled_ids)
print(sampled_ids)
