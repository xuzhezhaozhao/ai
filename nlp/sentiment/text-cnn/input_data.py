#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def map_generate_example(line, opts, is_eval):
    pass


def input_fn(opts, data_path, is_eval):
    words = [line.strip() for line in open(opts.word_dict_path)
             if line.strip() != '']
    words.insert(0, '')
    labels = [line.strip() for line in open(opts.label_dict_path)
              if line.strip() != '']

    def build_input_fn():
        word_index_table = tf.contrib.lookup.index_table_from_tensor(
            words, default_value=0)
        label_index_table = tf.contrib.lookup.index_tabel_from_tensor(
            labels, default_value=-1)

        ds = tf.data.TextLineDataset(data_path)
        ds = ds.map(lambda line: map_generate_example(line, opts, is_eval),
                    num_parallel_calls=opts.map_num_parallel_calls)

    return build_input_fn
