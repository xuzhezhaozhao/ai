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


def input_feature_columns(opts):
    """Return a tuple of feature_columns. One for train, one for predict."""

    records_column = tf.feature_column.numeric_column(
        key=model_keys.RECORDS_COL, shape=[opts.train_ws], dtype=tf.int32)
    predict_records_column = tf.feature_column.numeric_column(
        key=model_keys.RECORDS_COL, shape=[opts.predict_ws], dtype=tf.int32)

    if opts.age_feature_type == model_keys.AgeFeatureType.indicator:
        age_column, age_dim = age_indicator_column()
    elif opts.age_feature_type == model_keys.AgeFeatureType.numeric:
        age_column, age_dim = age_numeric_column()
    else:
        raise ValueError("Unsurpported age feature type.")

    gender_column, gender_dim = gender_indicator_column()

    user_features_dim = 0
    user_features_columns = []
    if opts.use_age_feature:
        user_features_dim += age_dim
        user_features_columns.append(age_column)
    if opts.use_gender_feature:
        user_features_dim += gender_dim
        user_features_columns.append(gender_column)

    return ([records_column], [predict_records_column],
            user_features_columns, user_features_dim)


def age_numeric_column():
    def normalizer_fn(x):
        return tf.concat([(x-30.0)/100.0, (x*x - 400.0)/10000.0], axis=1)
    age_dim = 2
    age_numeric_column = tf.feature_column.numeric_column(
        key=model_keys.AGE_COL, shape=[age_dim], dtype=tf.float32,
        normalizer_fn=normalizer_fn)

    return age_numeric_column, age_dim


def age_indicator_column():
    age = tf.feature_column.numeric_column(key=model_keys.AGE_COL,
                                           shape=[1],
                                           dtype=tf.float32)
    boundaries = [10, 13, 15, 18, 20, 24, 28, 35, 40]
    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=boundaries)
    age_indicator_column = tf.feature_column.indicator_column(age_buckets)
    age_dim = len(boundaries) + 1
    return age_indicator_column, age_dim


def gender_indicator_column():
    gender_vocabulary = [0, 1, 2]
    gender_dim = len(gender_vocabulary)
    gender_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key=model_keys.GENDER_COL, vocabulary_list=gender_vocabulary,
        default_value=0, dtype=tf.int64)
    gender_indicator_column = tf.feature_column.indicator_column(gender_column)

    return gender_indicator_column, gender_dim



def pack_fasttext_params(opts, is_eval):
    params = dict()
    params['train_data_path'] = opts.train_data_path
    params['dict_dir'] = opts.dict_dir
    params['ws'] = opts.train_ws
    params['lower_ws'] = opts.train_lower_ws
    params['min_count'] = opts.min_count
    params['t'] = opts.t
    params['verbose'] = opts.verbose
    params['min_count_label'] = opts.min_count_label
    params['label'] = opts.label
    params['ntargets'] = opts.ntargets
    params['sample_dropout'] = opts.sample_dropout

    params['use_user_features'] = opts.use_user_features
    params['user_features_file'] = opts.user_features_file

    if is_eval:
        params['t'] = 1.0
        params['ntargets'] = 1
        params['sample_dropout'] = 0
        params['ws'] = opts.predict_ws

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


def need_ntokens(opts):
    if (opts.optimizer_type == model_keys.OptimizerType.SGD
            and (opts.sgd_lr_decay_type
                 == model_keys.SGDLrDecayType.FASTTEXT_DECAY)):
        return True
    return False


def map_generate_example(line, opts, is_eval):
    """Map a single line to features tuple."""

    params = pack_fasttext_params(opts, is_eval)
    params['input'] = line
    params['use_saved_dict'] = True

    (records, labels, ntokens,
     age, gender) = custom_ops.fasttext_example_generate(**params)

    results = [records, labels]
    if need_ntokens(opts):
        results.append(ntokens)

    if opts.use_user_features:
        results.append(age)
        results.append(gender)

    return tuple(results)


def flat_map_example(opts, x):
    """Deal with features tuple returned by map_generate_example.

    Returns:
        A 'Dataset'.
    """

    features_dict = {model_keys.RECORDS_COL: x[0]}
    labels = x[1]

    bias = 0
    if need_ntokens(opts):
        features_dict[model_keys.TOKENS_COL] = x[2]
        bias += 1

    if opts.use_user_features:
        features_dict[model_keys.AGE_COL] = x[2+bias]
        features_dict[model_keys.GENDER_COL] = x[3+bias]

    dataset = tf.data.Dataset.from_tensor_slices((features_dict, labels))
    return dataset


def train_input_fn(opts, skip_rows=0):
    if opts.train_data_format == model_keys.TrainDataFormatType.fasttext:
        return fasttext_train_input_fn(opts, skip_rows)
    elif opts.train_data_format == model_keys.TrainDataFormatType.tfrecord:
        return tfrecord_train_input_fn(opts)
    else:
        raise ValueError("Unsurpported train data format type '{}'".format(
            opts.train_data_format))


def fasttext_train_input_fn(opts, skip_rows=0):
    train_data_path = opts.train_data_path
    batch_size = opts.batch_size

    ds = tf.data.TextLineDataset(train_data_path).skip(skip_rows)
    # ref: https://stackoverflow.com/questions/47411383/parallel-threads-with-tensorflow-dataset-api-and-flat-map/47414078
    ds = ds.map(lambda line: map_generate_example(line, opts, False),
                num_parallel_calls=opts.map_num_parallel_calls)
    ds = ds.prefetch(opts.prefetch_size).flat_map(
        lambda *x: flat_map_example(opts, x))

    if opts.shuffle_batch:
        ds = ds.shuffle(buffer_size=opts.shuffle_size)
    ds = ds.batch(batch_size).repeat(opts.epoch)
    return ds


def parse_example(serialized, opts):
    example = tf.parse_single_example(
        serialized,
        features={
            'records': tf.FixedLenFeature([opts.train_ws], tf.int64),
            'label': tf.FixedLenFeature([opts.ntargets], tf.int64)
        }
    )
    return (example, example['label'])


def tfrecord_train_input_fn(opts):
    batch_size = opts.batch_size

    if opts.num_tfrecord_file > 1:
        files = []
        for seq in range(opts.num_tfrecord_file):
            files.append(opts.tfrecord_file + '.' + '{:03d}'.format(seq))
        ds = tf.data.TFRecordDataset(files)
    else:
        ds = tf.data.TFRecordDataset([opts.tfrecord_file])

    ds = ds.map(lambda x: parse_example(x, opts),
                num_parallel_calls=opts.map_num_parallel_calls)
    ds = ds.prefetch(opts.prefetch_size)
    if opts.shuffle_batch:
        ds = ds.shuffle(buffer_size=opts.shuffle_size)
    ds = ds.batch(batch_size).repeat(opts.epoch)
    return ds


def eval_input_fn(opts, skip_rows=0):
    eval_data_path = opts.eval_data_path
    batch_size = opts.batch_size

    ds = tf.data.TextLineDataset(eval_data_path).skip(skip_rows)
    ds = ds.map(lambda line: map_generate_example(line, opts, True),
                num_parallel_calls=opts.map_num_parallel_calls)
    ds = ds.prefetch(opts.prefetch_size).flat_map(
        lambda *x: flat_map_example(opts, x))
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

        if opts.use_user_features:
            feature_spec[model_keys.AGE_COL] = tf.FixedLenFeature(
                shape=[1], dtype=tf.float32)
            feature_spec[model_keys.GENDER_COL] = tf.FixedLenFeature(
                shape=[1], dtype=tf.int64)

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
