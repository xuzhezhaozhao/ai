#! /usr/bin/env python
# -*- coding=utf8 -*-

import tensorflow as tf
import threading


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


def thread_body(sess, op):
    print(sess.run(op))


print(sess.run(sampled_ids))

workers = []
for _ in xrange(4):
    worker = threading.Thread(target=thread_body, args=(sess, sampled_ids))
    worker.start()
    workers.append(worker)

for worker in workers:
    worker.join()
