#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

MODEL_DIR = './model_dir'
n_classes = 10000
dim = 100

""" get_variable 需要在 Saver 定义之前 """
embeddings = tf.get_variable(name='embeddings', shape=[n_classes, dim],
                             initializer=tf.zeros_initializer)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(embeddings.initializer)
    print("init embeddings = \n{}\n".format(sess.run(embeddings)))
    module_file = tf.train.latest_checkpoint(MODEL_DIR)
    print("checkpoint = {}".format(module_file))
    saver.restore(sess, module_file)
    print("restore embeddings = \n{}\n".format(sess.run(embeddings)))
