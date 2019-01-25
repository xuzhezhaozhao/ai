#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

IMAGE_SIZE = 64


def build_train_input_fn(opts, train_data_path):
    def train_input_fn():
        ds = create_image_dataset(train_data_path)
        num_parallel = opts.map_num_parallel_calls
        ds = ds.map(
            lambda filename: parse_function(filename, opts),
            num_parallel_calls=num_parallel)
        ds = ds.prefetch(opts.prefetch_size)
        if opts.shuffle_batch:
            ds = ds.shuffle(buffer_size=opts.shuffle_size)
        ds = ds.repeat().batch(opts.batch_size)

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


def parse_function(img_path, opts):
    image_string = tf.read_file(img_path)
    image_decoded = tf.image.decode_image(image_string, channels=None)
    image_decoded = image_decoded[:, :, :opts.nc]
    image_decoded.set_shape([None, None, opts.nc])
    image = tf.cast(image_decoded, tf.float32)

    img_size = opts.img_size
    smallest_side = tf.random_uniform(
        [],
        minval=img_size,
        maxval=img_size+1,
        dtype=tf.int32)
    smallest_side = tf.to_float(smallest_side)
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.to_float(height)
    width = tf.to_float(width)
    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    resized_image = tf.image.resize_images(image, [new_height, new_width])
    crop_image = tf.random_crop(resized_image, [img_size, img_size, opts.nc])
    norm_image = crop_image / 127.5 - 1.0

    return {'data': norm_image}


def invert_norm(x):
    return (x + 1.0) * 127.5
