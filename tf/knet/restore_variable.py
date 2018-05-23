#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

MODEL_DIR = './model_dir'
n_classes = 300000
dim = 100

""" get_variable 需要在 Saver 定义之前 """
embeddings = tf.get_variable('embeddings', shape=[n_classes, dim])
saver = tf.train.Saver()

with tf.Session() as sess:
    module_file = tf.train.latest_checkpoint(MODEL_DIR)
    print("checkpoint = {}".format(module_file))
    saver.restore(sess, module_file)
    result = sess.run(embeddings)
    print("restore embeddings = \n{}\n".format(result))

    print("details")
    for e in result:
        print(e)
