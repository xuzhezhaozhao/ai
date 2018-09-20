#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model_keys
import custom_ops


def input_feature_columns(opts):
    positive_records_col = tf.feature_column.numeric_column(
        key=model_keys.POSITIVE_RECORDS_COL,
        shape=[opts.train_ws],
        dtype=tf.int32)
    negative_records_col = tf.feature_column.numeric_column(
        key=model_keys.NEGATIVE_RECORDS_COL,
        shape=[opts.train_ws],
        dtype=tf.int32)
    targets_col = tf.feature_column.numeric_column(
        key=model_keys.TARGETS_COL,
        shape=[1],
        dtype=tf.int64)

    return (positive_records_col, negative_records_col, targets_col)


def map_generate_example(line, opts, is_eval):
    """Map a single line to features tuple."""

    out1, out2, out3, out4 = custom_ops.krank_input(
        input=line, feature_manager_path=opts.feature_manager_path,
        ws=opts.train_ws)
    return (out1, out2, out3, out4)


def flat_map_example(opts, x):
    """Deal with features tuple returned by map_generate_example.

    Returns:
        A 'Dataset'.
    """
    feature_dict = {
        model_keys.POSITIVE_RECORDS_COL: x[0],
        model_keys.NEGATIVE_RECORDS_COL: x[1],
        model_keys.TARGETS_COL: x[2],
    }
    labels = x[3]
    dataset = tf.data.Dataset.from_tensor_slices((feature_dict, labels))

    return dataset


def input_fn(opts, is_eval):
    train_data_path = opts.train_data_path
    batch_size = opts.batch_size

    ds = tf.data.TextLineDataset(train_data_path)
    ds = ds.map(lambda line: map_generate_example(line, opts, False),
                num_parallel_calls=opts.map_num_parallel_calls)
    ds = ds.prefetch(opts.prefetch_size).flat_map(
        lambda *x: flat_map_example(opts, x))

    if opts.shuffle_batch and not is_eval:
        ds = ds.shuffle(buffer_size=opts.shuffle_size)

    ds = ds.batch(batch_size)
    if not is_eval:
        ds = ds.repeat(opts.epoch)

    return ds


def build_serving_input_fn(opts):
    def serving_input_receiver_fn():
        """An input receiver that expects a serialized tf.Example.
        Note: Set serialized_tf_example shape as [None] to handle variable
        batch size
        """
        pass

    return serving_input_receiver_fn
