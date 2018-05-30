#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


RECORDS_COL = "records"
WORDS_COL = "words"

DICT_META = "dict_meta"
DICT_WORDS = "dict_words"
SAVED_DICT_BIN = "saved_dict.bin"

# change to absolute path when using tesla
# ROOT_OPS_PATH = '/cephfs/group/sng-im-sng-imappdev-tribe/zhezhaoxu/knet'
ROOT_OPS_PATH = ''
FASTTEXT_EXAMPLE_GENERATE_OPS_PATH = os.path.join(
    ROOT_OPS_PATH, 'fasttext_example_generate_ops.so')
OPENBLAS_TOP_K_OPS_PATH = os.path.join(
    ROOT_OPS_PATH, 'openblas_top_k_ops.so')

fasttext_example_generate_ops = tf.load_op_library(
    FASTTEXT_EXAMPLE_GENERATE_OPS_PATH)

openblas_top_k_ops = tf.load_op_library(OPENBLAS_TOP_K_OPS_PATH)
openblas_top_k = openblas_top_k_ops.openblas_top_k


def feature_columns(opts):
    my_feature_columns = []
    my_feature_columns.append(tf.feature_column.numeric_column(
        key=RECORDS_COL, shape=[opts.ws], dtype=tf.int32))
    return my_feature_columns


def init_dict(opts):
    if opts.use_saved_dict:
        return

    dummy1, dummy2 = fasttext_example_generate_ops.fasttext_example_generate(
        train_data_path=opts.train_data_path,
        input="",
        use_saved_dict=False,
        dict_dir=opts.dict_dir,
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
    with tf.Session() as sess:
        sess.run(dummy1)


def parse_dict_meta(opts):
    dict_meta = {}
    for line in open(os.path.join(opts.dict_dir, DICT_META)):
        tokens = line.strip().split('\t')
        if len(tokens) != 2:
            tf.logging.info("parse dict meta error line: {}".format(line))
            continue
        dict_meta[tokens[0]] = int(tokens[1])
    tf.logging.info("\ndict_meta = \n{}\n".format(dict_meta))
    return dict_meta


def generate_example(line, opts):
    """
    测试时需要使用 initializable iterator
        it = ds.make_initializable_iterator()
        sess.run(it.initializer)
        sess.run(it.get_next())
    """
    records, labels = fasttext_example_generate_ops.fasttext_example_generate(
        input=line,
        train_data_path=opts.train_data_path,
        use_saved_dict=True,
        dict_dir=opts.dict_dir,
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
        ({RECORDS_COL: records}, labels))
    return dataset


def train_input_fn(opts, skip_rows=0):
    train_data_path = opts.train_data_path
    batch_size = opts.batch_size

    ds = tf.data.TextLineDataset(train_data_path).skip(skip_rows)
    ds = ds.flat_map(lambda line: generate_example(line, opts))
    ds = ds.prefetch(opts.prefetch_size)
    ds = ds.repeat(opts.epoch).batch(batch_size)  # no shuffle
    return ds


def eval_input_fn(opts, skip_rows=0):
    eval_data_path = opts.eval_data_path
    batch_size = opts.batch_size

    ds = tf.data.TextLineDataset(eval_data_path).skip(skip_rows)
    ds = ds.flat_map(lambda line: generate_example(line, opts))
    ds = ds.prefetch(opts.prefetch_size)
    ds = ds.batch(batch_size)
    return ds


def build_serving_input_fn(opts):
    words_feature = tf.FixedLenFeature(shape=[opts.ws], dtype=tf.string)

    def serving_input_receiver_fn():
        """An input receiver that expects a serialized tf.Example.
        Note: Set serialized_tf_example shape as [None] to handle variable
        batch size
        """
        feature_spec = {
            WORDS_COL: words_feature,
        }

        serialized_tf_example = tf.placeholder(dtype=tf.string,
                                               shape=[None],
                                               name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)

        words = [line.strip() for line in
                 open(os.path.join(opts.dict_dir, DICT_WORDS))
                 if line.strip() != '']
        words.insert(0, '')

        tf.logging.info(
            "serving_input_receiver_fn words size = {}".format(len(words)))

        table = tf.contrib.lookup.index_table_from_tensor(
            mapping=words, default_value=0)
        ids = table.lookup(features[WORDS_COL])
        features[RECORDS_COL] = ids

        return tf.estimator.export.ServingInputReceiver(features,
                                                        receiver_tensors)

    return serving_input_receiver_fn
