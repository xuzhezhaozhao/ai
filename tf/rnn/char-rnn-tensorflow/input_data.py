#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def parse_txt(data_path):
    text = open(data_path).read()
    tf.logging.info('Length of text: {} characters'.format(len(text)))
    vocab = sorted(set(text))
    tf.logging.info('{} unique characters'.format(len(vocab)))

    # Creating a mapping from unique characters to indices
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])

    return text_as_int


def build_train_input_fn(opts, data_path):
    def train_input_fn():
        text_as_int = parse_txt(data_path)
        text_as_int = tf.data.Dataset.from_tensor_slices(text_as_int)
        chunks = text_as_int.batch(opts.seq_length + 1, drop_remainder=True)

        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        dataset = chunks.map(split_input_target,
                             num_parallel_calls=opts.map_num_parallel_calls)
        if opts.shuffle_batch:
            dataset = dataset.shuffle(opts.shuffle_size)
        dataset = dataset.batch(opts.batch_size, drop_remainder=True)
        dataset = dataset.repeat()

        return dataset

    return train_input_fn


def build_eval_input_fn(opts, data_path):
    def eval_input_fn():
        pass
    return eval_input_fn
