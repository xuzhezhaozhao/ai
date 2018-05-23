#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


FASTTEXT_MODEL_PATH = '../ops/ftlib/fasttext_example_generate_ops.so'
fasttext_model = tf.load_op_library(FASTTEXT_MODEL_PATH)


def generate_example(line, options):
    """
    测试时需要使用 initializable iterator
        it = ds.make_initializable_iterator()
        sess.run(it.initializer)
        sess.run(it.get_next())
    """
    (records, labels) = fasttext_model.fasttext_example_generate(
        input=line,
        train_data_path=options.train_data_path,
        dim=options.dim,
        maxn=options.maxn,
        minn=options.minn,
        word_ngrams=options.word_ngrams,
        bucket=options.bucket,
        ws=options.ws,
        min_count=options.min_count,
        t=options.t,
        verbose=options.verbose,
        min_count_label=options.min_count_label,
        label=options.label,
        seed=options.seed
    )
    dataset = tf.data.Dataset.from_tensor_slices(
        ({"records": records}, labels)
    )
    return dataset


def train_input_fn(options, skip_rows=0):
    train_data_path = options.train_data_path
    batch_size = options.batch_size

    ds = tf.data.TextLineDataset(train_data_path).skip(skip_rows)
    ds = ds.flat_map(lambda line: generate_example(line, options))
    ds = ds.shuffle(1000).repeat().batch(batch_size)
    return ds
