#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from options import Options
from model import my_model

import argparse
import tensorflow as tf

import input_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

COLUMN_NAMES = ["records"]


def feature_default():
    return tf.FixedLenFeature(shape=[5], dtype=tf.int64)


feature_spec = {
    'records': feature_default(),
}


def serving_input_receiver_fn():
    """An input receiver that expects a serialized tf.Example.
    Note: Set serialized_tf_example shape as [None] to handle variable
    batch size
    """
    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                           shape=[None],
                                           name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}
    raw_features = tf.parse_example(serialized_tf_example, feature_spec)

    features = raw_features
    # Do anything to raw_features ...
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def main(argv):
    args = parser.parse_args(argv[1:])
    opts = Options()
    opts.batch_size = args.batch_size
    opts.train_data_path = '../ops/ftlib/train_data.in'
    opts.lr = 1.0
    opts.dim = 100
    opts.maxn = 0
    opts.minn = 0
    opts.word_ngrams = 1
    opts.bucket = 2000000
    opts.ws = 20
    opts.min_count = 50
    opts.t = 0.0001
    opts.verbose = 1
    opts.min_count_label = 1
    opts.label = "__label__"
    opts.batch_size = 64
    opts.num_sampled = 10
    opts.max_train_steps = None
    opts.epoch = 25

    # Feature columns describe how to use the input.
    my_feature_columns = []
    my_feature_columns.append(tf.feature_column.numeric_column(
        key="records", shape=[opts.ws], dtype=tf.int32))

    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        model_dir="model_dir",
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [128, 64],
            # The model must choose between 3 classes.
            'n_classes': 300000,  # TODO
            'embedding_dim': opts.dim,
            'learning_rate': opts.lr,
            'num_sampled': opts.num_sampled
        })

    classifier.train(input_fn=lambda: input_data.train_input_fn(opts),
                     max_steps=opts.max_train_steps)
    classifier.export_savedmodel(
        "export_model_dir",
        serving_input_receiver_fn=serving_input_receiver_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
