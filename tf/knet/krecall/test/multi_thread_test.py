#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import threading


sess = None
c = None


def get_constant():
    with tf.variable_scope("constant", reuse=tf.AUTO_REUSE):
        c = tf.get_variable("c1", initializer=tf.constant([0, 0, 0]))
    return c


def thread_target_fn(id):
    global sess, c

    op = tf.assign_add(c, [id, id, id])
    op = sess.run(op)

    print("thread {}, op = {}".format(id, op))


def main(argv):
    global sess, c
    sess = tf.Session()
    c = get_constant()
    sess.run(tf.global_variables_initializer())

    th1 = threading.Thread(target=thread_target_fn,
                           args=(1, ))
    th2 = threading.Thread(target=thread_target_fn,
                           args=(2, ))
    th1.start()
    th2.start()

    th1.join()
    th2.join()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
