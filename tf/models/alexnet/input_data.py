#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


def parse_line(line, opts):
    img_path, label = line.split(' ')
    one_hot = tf.one_hot(label, opts.num_classes)

    # load and preprocess the image
    img_string = tf.read_file(img_path)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])

    # Data augmentation comes here.
    img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

    # RGB -> BGR
    img_bgr = img_centered[:, :, ::-1]

    return img_bgr, one_hot


def train_input_fn(opts, skip_rows=0):
    ds = tf.data.TextLineDataset(opts.train_data_path).skip(skip_rows)
    ds = ds.map(lambda line: parse_line(line, opts),
                num_parallel_calls=opts.map_num_parallel_calls)
    ds = ds.prefetch(opts.prefetch_size)
    if opts.shuffle_batch:
        ds = ds.shuffle(buffer_size=opts.shuffle_size)
    ds = ds.batch(opts.batch_size).repeat(opts.epoch)
    return ds


def eval_input_fn(opts, skip_rows=0):
    ds = tf.data.TextLineDataset(opts.train_data_path).skip(skip_rows)
    ds = ds.map(lambda line: parse_line(line, opts),
                num_parallel_calls=opts.map_num_parallel_calls)
    ds = ds.prefetch(opts.prefetch_size)
    ds = ds.batch(opts.batch_size)
    return ds


def build_serving_input_fn(opts):
    def serving_input_receiver_fn():
        pass

    return serving_input_receiver_fn
