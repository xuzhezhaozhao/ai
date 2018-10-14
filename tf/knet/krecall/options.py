#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Options(object):
    def __init__(self):
        pass

    def list_all_member(self):
        tf.logging.info("\tOptions:")
        for name, value in vars(self).items():
            tf.logging.info('\t{} = {}'.format(name, value))
