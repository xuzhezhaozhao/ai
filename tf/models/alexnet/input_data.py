#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import model_keys


def parse_line(img_path, label, opts):

    one_hot = tf.one_hot(label, opts.num_classes)

    # load and preprocess the image
    img_string = tf.read_file(img_path)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])

    IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
    img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

    # RGB -> BGR
    img_bgr = img_centered[:, :, ::-1]

    return {model_keys.DATA_COL: img_bgr}, one_hot


def read_txt_file(txt_file):
    """Read the content of the text file and store it into lists."""

    img_paths = []
    labels = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(' ')
            img_paths.append(items[0])
            labels.append(int(items[1]))

    return img_paths, labels


def shuffle_lists(img_paths, labels):
    """Conjoined shuffling of the list of paths and labels."""

    path = img_paths
    labels = labels
    permutation = np.random.permutation(len(labels))
    shuffle_img_paths = []
    shuffle_labels = []
    for i in permutation:
        shuffle_img_paths.append(path[i])
        shuffle_labels.append(labels[i])

    return shuffle_img_paths, shuffle_labels


def train_input_fn(opts):
    img_paths, img_labels = read_txt_file(opts.train_data_path)
    img_paths, img_labels = shuffle_lists(img_paths, img_labels)

    img_paths = tf.convert_to_tensor(img_paths, dtype=tf.string)
    labels = tf.convert_to_tensor(img_labels, dtype=tf.int32)

    # create dataset
    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))

    ds = ds.map(lambda filename, label: parse_line(filename, label, opts),
                num_parallel_calls=opts.map_num_parallel_calls)

    ds = ds.prefetch(opts.prefetch_size)
    if opts.shuffle_batch:
        ds = ds.shuffle(buffer_size=opts.shuffle_size)
    ds = ds.batch(opts.batch_size).repeat(opts.epoch)

    return ds


def eval_input_fn(opts):
    img_paths, img_labels = read_txt_file(opts.eval_data_path)
    img_paths = tf.convert_to_tensor(img_paths, dtype=tf.string)
    labels = tf.convert_to_tensor(img_labels, dtype=tf.int32)

    # create dataset
    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    ds = ds.map(lambda filename, label: parse_line(filename, label, opts),
                num_parallel_calls=opts.map_num_parallel_calls)

    ds = ds.prefetch(opts.prefetch_size)
    ds = ds.batch(opts.batch_size)
    return ds


def build_serving_input_fn(opts):
    def serving_input_receiver_fn():
        feature_spec = {
            model_keys.DATA_COL: tf.FixedLenFeature(shape=[227, 227, 3],
                                                    dtype=tf.float32)
        }

        serialized_tf_example = tf.placeholder(dtype=tf.string,
                                               shape=[None],
                                               name='input_example_tensor')

        receiver_tensors = {'examples': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)

        return tf.estimator.export.ServingInputReceiver(features,
                                                        receiver_tensors)

    return serving_input_receiver_fn
