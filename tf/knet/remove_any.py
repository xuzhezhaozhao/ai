#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def main(argv):
    tf.logging.info("TO delete files: {}".format(argv[1:]))
    for filename in argv[1:]:
        if tf.gfile.Exists(filename):
            if tf.gfile.IsDirectory(filename):
                tf.logging.info("delete dir '{}' ...".format(filename))
                tf.gfile.DeleteRecursively(filename)
            else:
                tf.logging.info("delete file '{}' ...".format(filename))
                tf.gfile.Remove(filename)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
