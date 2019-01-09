#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

text_cnn_input_ops = tf.load_op_library('./lib/text_cnn_input_ops.so')


def parse_function(text, opts, is_predict):
    words = [word.strip() for word in open(opts.word_dict_path)
             if word.strip() != '']
    words.insert(0, '')
    labels = [label.strip() for label in open(opts.label_dict_path)
              if label.strip() != '']
    label_weights = map(float, opts.label_weights)
    assert len(label_weights) == len(labels)
    word_ids, label = text_cnn_input_ops.text_cnn_input(
        input=text,
        word_dict=tf.make_tensor_proto(words),
        label_dict=tf.make_tensor_proto(labels),
        label_str=opts.label_str,
        max_length=opts.max_length)

    label_weights = tf.convert_to_tensor(label_weights)
    weight = tf.nn.embedding_lookup(label_weights, label)
    if is_predict:
        return {'data': word_ids}
    else:
        return {'data': word_ids, 'label_weights': weight}, label


def build_train_input_fn(opts, data_path):

    def train_input_fn():
        ds = tf.data.TextLineDataset(data_path)
        ds = ds.map(lambda line: parse_function(line, opts, is_predict=False),
                    num_parallel_calls=opts.map_num_parallel_calls)
        ds = ds.prefetch(opts.prefetch_size)
        if opts.shuffle_batch:
            ds = ds.shuffle(buffer_size=opts.shuffle_size)
        ds = ds.repeat(opts.epoch)
        ds = ds.batch(opts.batch_size)

        return ds

    return train_input_fn


def build_eval_input_fn(opts, data_path):

    def eval_input_fn():
        ds = tf.data.TextLineDataset(data_path)
        ds = ds.map(lambda line: parse_function(line, opts, is_predict=False),
                    num_parallel_calls=opts.map_num_parallel_calls)
        ds = ds.prefetch(opts.prefetch_size)
        ds = ds.batch(opts.eval_batch_size)

        return ds

    return eval_input_fn


def build_predict_input_fn(opts, data_path):

    def predict_input_fn():
        ds = tf.data.TextLineDataset(data_path)
        ds = ds.map(lambda line: parse_function(line, opts, is_predict=True),
                    num_parallel_calls=opts.map_num_parallel_calls)
        ds = ds.prefetch(opts.prefetch_size)
        ds = ds.batch(opts.eval_batch_size)

        return ds

    return predict_input_fn
