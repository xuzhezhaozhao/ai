
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import tensorflow as tf

string_split_v2 = tf.load_op_library(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'string_split_v2_ops.so'))


def main(_):
    with tf.Graph().as_default(), tf.Session() as sess:
        x = tf.constant(["a b c", "aa bb cc dd"])
        x = string_split_v2.string_split_v2(x, " ")
        print(sess.run(x))


if __name__ == "__main__":
    tf.app.run()
