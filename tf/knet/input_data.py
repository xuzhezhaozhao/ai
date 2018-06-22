#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np

import model_keys
import custom_ops


def feature_columns(opts):
    """Return a tuple of feature_columns. One for train, one for predict."""

    my_feature_columns = []
    predict_feature_columns = []
    my_feature_columns.append(tf.feature_column.numeric_column(
        key=model_keys.RECORDS_COL, shape=[opts.train_ws], dtype=tf.int32))
    predict_feature_columns.append(tf.feature_column.numeric_column(
        key=model_keys.RECORDS_COL, shape=[opts.predict_ws], dtype=tf.int32))
    return my_feature_columns, predict_feature_columns


def pack_fasttext_params(opts, is_eval):
    params = dict()
    params['train_data_path'] = opts.train_data_path
    params['dict_dir'] = opts.dict_dir
    params['ws'] = opts.train_ws
    params['min_count'] = opts.min_count
    params['t'] = opts.t
    params['verbose'] = opts.verbose
    params['min_count_label'] = opts.min_count_label
    params['label'] = opts.label
    params['ntargets'] = opts.ntargets
    params['sample_dropout'] = opts.sample_dropout

    if is_eval:
        params['t'] = 1.0
        params['ntargets'] = 1
        params['sample_dropout'] = 0

    return params


def init_dict(opts):
    if opts.use_saved_dict:
        return

    params = pack_fasttext_params(opts, False)
    params['input'] = ''
    params['use_saved_dict'] = False
    dummy = custom_ops.fasttext_example_generate(**params)
    with tf.Session() as sess:
        sess.run(dummy)


def parse_dict_meta(opts):
    dict_meta = {}
    for line in open(os.path.join(opts.dict_dir, model_keys.DICT_META)):
        tokens = line.strip().split('\t')
        if len(tokens) != 2:
            tf.logging.info("parse dict meta error line: {}".format(line))
            continue
        dict_meta[tokens[0]] = int(tokens[1])
    tf.logging.info("\ndict_meta = \n{}\n".format(dict_meta))
    return dict_meta


def generate_example(line, opts, is_eval):
    """
    测试时需要使用 initializable iterator
        it = ds.make_initializable_iterator()
        sess.run(it.initializer)
        sess.run(it.get_next())
    """

    params = pack_fasttext_params(opts, is_eval)
    params['input'] = line
    params['use_saved_dict'] = True

    records, labels, tokens = custom_ops.fasttext_example_generate(**params)
    dataset = tf.data.Dataset.from_tensor_slices(
        ({model_keys.RECORDS_COL: records, model_keys.TOKENS_COL: tokens},
         labels))
    return dataset


def train_input_fn(opts, skip_rows=0):
    train_data_path = opts.train_data_path
    batch_size = opts.batch_size * opts.num_in_graph_replication

    ds = tf.data.TextLineDataset(train_data_path).skip(skip_rows)
    ds = ds.flat_map(lambda line: generate_example(line, opts, False))
    ds = ds.prefetch(opts.prefetch_size)
    if opts.shuffle_batch:
        ds = ds.shuffle(buffer_size=opts.prefetch_size)
    ds = ds.batch(batch_size).repeat(opts.epoch)
    return ds


def multi_thread_train_input_fn(opts, skip_rows=0):
    batch_size = opts.batch_size * opts.num_in_graph_replication
    inputs = []

    for i in range(opts.threads):
        suf = '.{:02d}'.format(i)
        train_data_path = opts.train_data_path + suf

        ds = tf.data.TextLineDataset(train_data_path).skip(skip_rows)
        ds = ds.flat_map(lambda line: generate_example(line, opts, False))
        ds = ds.prefetch(opts.prefetch_size)
        if opts.shuffle_batch:
            ds = ds.shuffle(buffer_size=opts.prefetch_size)
        ds = ds.batch(batch_size).repeat(opts.epoch)
        inputs.append(ds)

    return inputs


def eval_input_fn(opts, skip_rows=0):
    eval_data_path = opts.eval_data_path
    batch_size = opts.batch_size

    ds = tf.data.TextLineDataset(eval_data_path).skip(skip_rows)
    ds = ds.flat_map(lambda line: generate_example(line, opts, True))
    ds = ds.prefetch(opts.prefetch_size)
    ds = ds.batch(batch_size)
    return ds


def train_random_numpy_input_fn(opts):
    """Generate dummy train examples for performence profile, see cpu usage."""

    n = 1000000
    examples = np.random.randint(1, 100, [n, opts.train_ws])
    labels = np.random.randint(1, 100, [n, opts.ntargets])

    return tf.estimator.inputs.numpy_input_fn(
        x={model_keys.RECORDS_COL: examples},
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
            model_keys.WORDS_COL: tf.FixedLenFeature(shape=[opts.receive_ws],
                                                     dtype=tf.string)
        }

        serialized_tf_example = tf.placeholder(dtype=tf.string,
                                               shape=[None],
                                               name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)

        words = [line.strip() for line in
                 open(os.path.join(opts.dict_dir, model_keys.DICT_WORDS))
                 if line.strip() != '']
        words.insert(0, '')
        tf.logging.info(
            "serving_input_receiver_fn words size = {}".format(len(words)))
        ids, num_in_dict = custom_ops.dict_lookup(
            input=features[model_keys.WORDS_COL],
            dict=tf.make_tensor_proto(words),
            output_ws=opts.predict_ws)
        features[model_keys.RECORDS_COL] = ids
        features[model_keys.NUM_IN_DICT_COL] = num_in_dict

        return tf.estimator.export.ServingInputReceiver(features,
                                                        receiver_tensors)

    return serving_input_receiver_fn
