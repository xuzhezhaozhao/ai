#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from preprocessing import preprocessing_factory
import model_keys

DATA_COL = model_keys.DATA_COL


def read_txt_file(txt_file, has_label=True):
    """Read the content of the text file and store it into lists."""

    img_paths = []
    labels = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            items = line.split(' ')
            img_paths.append(items[0])
            if has_label:
                labels.append(int(items[1]))

    if has_label:
        return img_paths, labels
    else:
        return img_paths


def shuffle_lists(img_paths, labels):
    """Conjoined shuffling of the list of paths and labels."""

    tf.logging.info("Shuffle images ...")

    path = img_paths
    labels = labels
    permutation = np.random.permutation(len(labels))
    shuffle_img_paths = []
    shuffle_labels = []
    for i in permutation:
        shuffle_img_paths.append(path[i])
        shuffle_labels.append(labels[i])

    tf.logging.info("Shuffle images done, first one is '{}'."
                    .format(shuffle_img_paths[0]))

    return shuffle_img_paths, shuffle_labels


def create_image_dataset(data_path, has_label=True):
    if has_label:
        img_paths, img_labels = read_txt_file(data_path, has_label=True)
        img_paths, img_labels = shuffle_lists(img_paths, img_labels)
        img_paths = tf.convert_to_tensor(img_paths, dtype=tf.string)
        labels = tf.convert_to_tensor(img_labels, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    else:
        img_paths = read_txt_file(data_path, has_label=False)
        img_paths = tf.convert_to_tensor(img_paths, dtype=tf.string)

        ds = tf.data.Dataset.from_tensor_slices((img_paths))

    return ds


def parse_function(img_path, label, is_training, opts, image_size):
    image_string = tf.read_file(img_path)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    image_decoded.set_shape([None, None, 3])
    image = tf.cast(image_decoded, tf.float32)

    preprocessing_name = opts.preprocess_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=is_training)

    image = image_preprocessing_fn(image, image_size, image_size)

    if label is None:
        return {DATA_COL: image}
    else:
        label = tf.one_hot(label, opts.num_classes)
        return {DATA_COL: image}, label


def build_train_input_fn(opts, data_path):
    def train_input_fn():
        ds = create_image_dataset(data_path, has_label=True)
        ds = ds.map(
            lambda filename, label: parse_function(
                filename, label, True, opts, opts.train_image_size),
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
        ds = create_image_dataset(data_path, has_label=True)
        ds = ds.map(
            lambda filename, label: parse_function(
                filename, label, False, opts, opts.inference_image_size),
            num_parallel_calls=opts.map_num_parallel_calls)
        ds = ds.prefetch(opts.prefetch_size)
        ds = ds.batch(opts.batch_size)
        return ds
    return eval_input_fn


def build_predict_input_fn(opts, data_path):
    def predict_input_fn():
        ds = create_image_dataset(data_path, has_label=False)
        ds = ds.map(
            lambda filename: parse_function(
                filename, None, False, opts, opts.inference_image_size),
            num_parallel_calls=opts.map_num_parallel_calls)
        ds = ds.prefetch(opts.prefetch_size)
        ds = ds.batch(opts.batch_size)
        return ds
    return predict_input_fn
