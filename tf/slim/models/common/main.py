#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

from common import train
from common import args_parser


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'


def main(_):
    D = args_parser.opts.flag_values_dict()
    tf.logging.info("FLAGS: ")
    for key in D:
        tf.logging.info('{} = {}'.format(key, D[key]))
    train.train(args_parser.opts, export=False)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
