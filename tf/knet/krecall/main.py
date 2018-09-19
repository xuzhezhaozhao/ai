#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import args_parser
import train


def main(argv):
    opts = args_parser.parse(argv)
    train.train(opts, export=True)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
