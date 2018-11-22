#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


# change to absolute path when using tesla
# ROOT_OPS_PATH = '/cephfs/group/sng-im-sng-imappdev-tribe/zhezhaoxu/knet'
ROOT_OPS_PATH = 'lib/'
FASTTEXT_EXAMPLE_GENERATE_OPS_PATH = os.path.join(
    ROOT_OPS_PATH, 'fasttext_example_generate_ops.so')
# OPENBLAS_TOP_K_OPS_PATH = os.path.join(ROOT_OPS_PATH, 'openblas_top_k_ops.so')
DICT_LOOKUP_OPS_PATH = os.path.join(ROOT_OPS_PATH, 'dict_lookup_ops.so')
FASTTEXT_NEGATIVE_SAMPLER_OPS_PATH = os.path.join(ROOT_OPS_PATH, 'fasttext_negative_sampler_ops.so')

fasttext_example_generate_ops = tf.load_op_library(
    FASTTEXT_EXAMPLE_GENERATE_OPS_PATH)
fasttext_example_generate = fasttext_example_generate_ops.fasttext_example_generate

# openblas_top_k_ops = tf.load_op_library(OPENBLAS_TOP_K_OPS_PATH)
# openblas_top_k = openblas_top_k_ops.openblas_top_k

dict_lookup_ops = tf.load_op_library(DICT_LOOKUP_OPS_PATH)
dict_lookup = dict_lookup_ops.dict_lookup

fasttext_negative_sampler_ops = tf.load_op_library(FASTTEXT_NEGATIVE_SAMPLER_OPS_PATH)
fasttext_negative_sampler = fasttext_negative_sampler_ops.fasttext_negative_sampler
