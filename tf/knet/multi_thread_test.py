#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import threading
import time


main_thread_scope = tf.VariableScope(tf.AUTO_REUSE, "main_thread_scope")


def get_constant(captured_scope):
    with tf.variable_scope(captured_scope, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("constant", reuse=tf.AUTO_REUSE):
            c = tf.get_variable("c1", initializer=tf.constant([1, 1, 1]))
    return c


def thread_target_fn(id, captured_scope):
    with tf.Session() as sess:
        c = get_constant(captured_scope)
        if id == 1:
            sess.run(tf.global_variables_initializer())
        print("thread {} init OK".format(id))
        time.sleep(1.0)
        print("thread {}, name = {}".format(id, c.name))

        op = tf.assign_add(c, [id, id, id])
        op = sess.run(op)
        c = sess.run(c)

        print("thread {}, op = {}".format(id, op))
        print("thread {}, c = {}".format(id, c))


def main(argv):
    th1 = threading.Thread(target=thread_target_fn,
                           args=(1, main_thread_scope))
    th1.start()
    th1.join()

    th2 = threading.Thread(target=thread_target_fn,
                           args=(2, main_thread_scope))
    th2.start()
    th2.join()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
