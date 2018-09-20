#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

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

    (positive_records, negative_records,
     targets, labels) = custom_ops.krank_input(
        input=line, feature_manager_path=opts.feature_manager_path,
        ws=opts.train_ws)

    return (positive_records, negative_records, targets, labels)


def flat_map_example(opts, x):
    """Deal with features tuple returned by map_generate_example.

    Returns:
        A 'Dataset'.
    """
    feature_dict = {
        model_keys.POSITIVE_RECORDS_COL: x[0],
        model_keys.NEGATIVE_RECORDS_COL: x[1],
        model_keys.TARGETS_COL: x[2]
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


def train_random_numpy_input_fn(opts):
    """Generate dummy train examples for performence profile, see cpu usage."""

    n = 100000
    positive_records = np.random.randint(0, opts.num_rowkey, [n, opts.train_ws])
    negative_records = np.random.randint(0, opts.num_rowkey, [n, opts.train_ws])
    targets = np.random.randint(1, 100, [n, 1])
    labels = np.random.random([n, 1]).astype(np.float32)

    return tf.estimator.inputs.numpy_input_fn(
        x={
            model_keys.POSITIVE_RECORDS_COL: positive_records,
            model_keys.NEGATIVE_RECORDS_COL: negative_records,
            model_keys.TARGETS_COL: targets
        },
        y=labels,
        batch_size=opts.batch_size,
        num_epochs=opts.epoch,
        shuffle=True
    )


def build_serving_input_fn(opts):
    def serving_input_receiver_fn():
        """An input receiver that expects a serialized tf.Example.
        Note: Set serialized_tf_example shape as [None] to handle variable
        batch size
        """
        feature_spec = {
            model_keys.POSITIVE_RECORDS_COL: tf.FixedLenFeature(
                shape=[opts.train_ws], dtype=tf.string),
            model_keys.NEGATIVE_RECORDS_COL: tf.FixedLenFeature(
                shape=[opts.train_ws], dtype=tf.string),
            model_keys.TARGETS_COL: tf.FixedLenFeature(
                shape=[1], dtype=tf.string)
        }

        serialized_tf_example = tf.placeholder(dtype=tf.string,
                                               shape=[None],
                                               name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)
        # TODO(zhezhaoxu) many things to do

    return serving_input_receiver_fn
