
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import threading
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

fasttext_model = tf.load_op_library(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'fasttext_ops.so'))

flags = tf.app.flags
flags.DEFINE_string("train_data", None, "train data file")

FLAGS = flags.FLAGS


def main(_):
    """Train a fasttext model."""
    (vacab_word,
     vacab_freq,
     words_per_epoch,
     current_epoch,
     total_words_processed,
     examples,
     labels,
     valid_lengths) = fasttext_model.fasttext(train_data=FLAGS.train_data,
                                              batch_size=10)

    result = dict()
    with tf.Session() as sess:
        (result['vacab_word'],
         result['vacab_freq'],
         result['words_per_epoch'],
         result['current_epoch'],
         result['total_words_processed'],
         result['examples'], result['labels'],
         result['valid_lengths']) = sess.run([vacab_word, vacab_freq,
                                             words_per_epoch,
                                             current_epoch,
                                             total_words_processed,
                                             examples,
                                             labels,
                                             valid_lengths])
        for key in result:
            print("{}: {}".format(key, result[key]))


if __name__ == "__main__":
    tf.app.run()
