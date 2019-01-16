#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

INPUT_SHAPE = [256, 256, 3]


def build_train_input_fn(opts, train_data_path):
    def train_input_fn():
        ds = create_image_dataset(train_data_path)
        num_parallel = opts.map_num_parallel_calls
        ds = ds.map(
            lambda filename: parse_function(filename, opts),
            num_parallel_calls=num_parallel)
        ds = ds.map(
            lambda image: train_preprocess(image, opts),
            num_parallel_calls=num_parallel)
        ds = ds.prefetch(opts.prefetch_size)
        if opts.shuffle_batch:
            ds = ds.shuffle(buffer_size=opts.shuffle_size)
        ds = ds.batch(opts.batch_size)

        return ds
    return train_input_fn


def create_image_dataset(data_path):
    img_paths = read_txt_file(data_path)
    img_paths = tf.convert_to_tensor(img_paths, dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices((img_paths))
    return ds


def read_txt_file(txt_file):
    """Read the content of the text file and store it into lists."""

    img_paths = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            items = line.split(' ')
            img_paths.append(items[0])
        return img_paths


def train_preprocess(image, opts):
    resized_image = tf.image.resize_images(image, INPUT_SHAPE[:-1])
    return {'data': resized_image}


def parse_function(img_path, opts):
    image_string = tf.read_file(img_path)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    image_decoded.set_shape([None, None, 3])
    image = tf.cast(image_decoded, tf.float32)
    return image
