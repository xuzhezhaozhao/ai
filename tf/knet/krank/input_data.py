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
         input=line,
         feature_manager_path=opts.feature_manager_path,
         ws=opts.train_ws,
         num_evaluate_target_per_line=opts.num_evaluate_target_per_line,
         log_per_lines=opts.log_per_lines,
         is_eval=is_eval)

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


def input_fn(opts, data_path, is_eval, epoch=1):
    def build_input_fn():
        batch_size = opts.eval_batch_size if is_eval else opts.batch_size

        with tf.name_scope("input_fn"):
            ds = tf.data.TextLineDataset(data_path)
            ds = ds.map(lambda line: map_generate_example(line, opts, is_eval),
                        num_parallel_calls=opts.map_num_parallel_calls)
            ds = ds.prefetch(opts.prefetch_size).flat_map(
                lambda *x: flat_map_example(opts, x))

            if opts.shuffle_batch and not is_eval:
                ds = ds.shuffle(buffer_size=opts.shuffle_size,
                                seed=np.random.randint(100000))

            if not is_eval:
                ds = ds.repeat(epoch)

            ds = ds.batch(batch_size)

        return ds

    return build_input_fn


def train_random_numpy_input_fn(opts):
    """Generate dummy train examples for performence profile, see cpu usage."""

    n = 100000
    positive_records = np.random.randint(0, opts.num_rowkey,
                                         [n, opts.train_ws])
    negative_records = np.random.randint(0, opts.num_rowkey,
                                         [n, opts.train_ws])
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
            'watched_rowkeys': tf.FixedLenFeature(
                shape=[opts.inference_actions_len], dtype=tf.string),
            'rinfo1': tf.FixedLenFeature(
                shape=[opts.inference_actions_len], dtype=tf.float32),
            'rinfo2': tf.FixedLenFeature(
                shape=[opts.inference_actions_len], dtype=tf.float32),
            'is_video': tf.FixedLenFeature(
                shape=[opts.inference_actions_len], dtype=tf.int64),
            'target_rowkeys': tf.FixedLenFeature(
                shape=[opts.inference_num_targets], dtype=tf.string),
            'num_targets': tf.FixedLenFeature(shape=[1], dtype=tf.int64)
        }

        serialized_tf_example = tf.placeholder(dtype=tf.string,
                                               shape=[None],
                                               name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)

        f = open(opts.feature_manager_path, 'rb')
        fe = f.read()
        (positive_records, negative_records,
         targets, is_target_in_dict,
         num_positive, num_negative) = custom_ops.krank_predict_input(
             watched_rowkeys=features['watched_rowkeys'],
             rinfo1=features['rinfo1'],
             rinfo2=features['rinfo2'],
             target_rowkeys=features[model_keys.TARGET_ROWKEYS_COL],
             num_targets=features['num_targets'],
             is_video=features['is_video'],
             feature_manager=fe,
             ws=opts.train_ws)
        features[model_keys.POSITIVE_RECORDS_COL] = positive_records
        features[model_keys.NEGATIVE_RECORDS_COL] = negative_records
        features[model_keys.TARGETS_COL] = targets
        features[model_keys.IS_TARGET_IN_DICT_COL] = is_target_in_dict
        features[model_keys.NUM_POSITIVE_COL] = num_positive
        features[model_keys.NUM_NEGATIVE_COl] = num_negative

        return tf.estimator.export.ServingInputReceiver(
            features, receiver_tensors)

    return serving_input_receiver_fn
