#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


tf.reset_default_graph()

MODEL_DIR = './model_dir'
n_classes = 10000
dim = 100

embeddings = tf.get_variable(name='embeddings', shape=[n_classes, dim])

saver = tf.train.Saver()

with tf.Session() as sess:
    module_file = tf.train.latest_checkpoint(MODEL_DIR)
    saver.restore(sess, module_file)
    embeddings = sess.run(embeddings)
    print("restore embeddings = \n{}\n".format(embeddings))
