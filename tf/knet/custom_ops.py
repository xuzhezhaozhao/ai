#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


# change to absolute path when using tesla
# ROOT_OPS_PATH = '/cephfs/group/sng-im-sng-imappdev-tribe/zhezhaoxu/knet'
ROOT_OPS_PATH = ''
FASTTEXT_EXAMPLE_GENERATE_OPS_PATH = os.path.join(
    ROOT_OPS_PATH, 'fasttext_example_generate_ops.so')
OPENBLAS_TOP_K_OPS_PATH = os.path.join(
    ROOT_OPS_PATH, 'openblas_top_k_ops.so')

fasttext_example_generate_ops = tf.load_op_library(
    FASTTEXT_EXAMPLE_GENERATE_OPS_PATH)
fasttext_example_generate = fasttext_example_generate_ops.fasttext_example_generate

openblas_top_k_ops = tf.load_op_library(OPENBLAS_TOP_K_OPS_PATH)
openblas_top_k = openblas_top_k_ops.openblas_top_k
