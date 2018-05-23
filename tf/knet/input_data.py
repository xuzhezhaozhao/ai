#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


FASTTEXT_MODEL_PATH = 'lib/fasttext_example_generate_ops.so'
fasttext_model = tf.load_op_library(FASTTEXT_MODEL_PATH)


def generate_example(line, opts):
    """
    测试时需要使用 initializable iterator
        it = ds.make_initializable_iterator()
        sess.run(it.initializer)
        sess.run(it.get_next())
    """
    (records, labels) = fasttext_model.fasttext_example_generate(
        input=line,
        train_data_path=opts.train_data_path,
        dim=opts.dim,
        maxn=opts.maxn,
        minn=opts.minn,
        word_ngrams=opts.word_ngrams,
        bucket=opts.bucket,
        ws=opts.ws,
        min_count=opts.min_count,
        t=opts.t,
        verbose=opts.verbose,
        min_count_label=opts.min_count_label,
        label=opts.label
    )
    dataset = tf.data.Dataset.from_tensor_slices(
        ({"records": records}, labels)
    )
    return dataset


def train_input_fn(opts, skip_rows=0):
    train_data_path = opts.train_data_path
    batch_size = opts.batch_size

    ds = tf.data.TextLineDataset(train_data_path).skip(skip_rows)
    ds = ds.flat_map(lambda line: generate_example(line, opts))
    ds = ds.prefetch(opts.prefetch_size)
    # no shuffle
    ds = ds.repeat(opts.epoch).batch(batch_size)
    return ds
