#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import model_keys

VGG_MEAN = [123.68, 116.779, 103.939]  # RGB
INPUT_SHAPE = [224, 224, 3]


class _dummy_label:
    pass


_DUMMY_LABEL = _dummy_label()


def read_txt_file(txt_file, has_label=True):
    """Read the content of the text file and store it into lists."""

    img_paths = []
    labels = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
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

    path = img_paths
    labels = labels
    permutation = np.random.permutation(len(labels))
    shuffle_img_paths = []
    shuffle_labels = []
    for i in permutation:
        shuffle_img_paths.append(path[i])
        shuffle_labels.append(labels[i])

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


# code modified from:
# https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c

# Standard preprocessing for VGG on ImageNet taken from here:
# https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py
# Also see the VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf


# Preprocessing (for both training and validation):
# (1) Decode the image from jpg/png/bmp format
# (2) Resize the image so its smaller side is 256 pixels long
def parse_function(img_path, label, opts):
    image_string = tf.read_file(img_path)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    image_decoded.set_shape([None, None, 3])
    image = tf.cast(image_decoded, tf.float32)

    smallest_side = tf.random_uniform(
        [],
        minval=opts.resize_side_min,
        maxval=opts.resize_side_max+1,
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

    if label is _DUMMY_LABEL:
        return resized_image
    else:
        return resized_image, label


# Preprocessing (for training)
# (3) Take a random 224x224 crop to the scaled image
# (4) Horizontally flip the image with probability 1/2
# (5) Substract the per color mean `VGG_MEAN`
# Note: we don't normalize the data here, as VGG was trained without
# normalization
# Note(zhezhaoxu): we add rgb to bgr transform
def train_preprocess(image, label, opts):
    crop_image = tf.random_crop(image, INPUT_SHAPE)             # (3)
    flip_image = tf.image.random_flip_left_right(crop_image)    # (4)
    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = flip_image - means                         # (5)
    bgr_image = centered_image[:, :, ::-1]  # RGB -> BGR

    label = tf.one_hot(label, opts.num_classes)
    return {model_keys.DATA_COL: bgr_image}, label


# Preprocessing (for validation)
# (3) Take a central 224x224 crop to the scaled image
# (4) Substract the per color mean `VGG_MEAN`
# Note: we don't normalize the data here, as VGG was trained without
# normalization
# Note(zhezhaoxu): we add multi scale image predict and rgb to bgr transform
def val_preprocess(image, label, opts):
    if opts.multi_scale_predict:
        if opts.inference_shape is not None:
            crop_image = tf.image.resize_image_with_crop_or_pad(
                image, opts.inference_shape[0], opts.inference_shape[1])
        else:
            crop_image = image  # do not crop image
    else:
        crop_image = tf.image.resize_image_with_crop_or_pad(
            image, INPUT_SHAPE[0], INPUT_SHAPE[1])

    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = crop_image - means
    bgr_image = centered_image[:, :, ::-1]  # RGB -> BGR

    if label is not _DUMMY_LABEL:
        label = tf.one_hot(label, opts.num_classes)
        return {model_keys.DATA_COL: bgr_image}, label
    else:
        return {model_keys.DATA_COL: bgr_image}  # for predict


def build_train_input_fn(opts):
    def train_input_fn():
        ds = create_image_dataset(opts.train_data_path, has_label=True)
        num_parallel = opts.map_num_parallel_calls
        ds = ds.map(
            lambda filename, label: parse_function(filename, label, opts),
            num_parallel_calls=num_parallel)
        ds = ds.map(
            lambda image, label: train_preprocess(image, label, opts),
            num_parallel_calls=num_parallel)
        ds = ds.prefetch(opts.prefetch_size)
        if opts.shuffle_batch:
            ds = ds.shuffle(buffer_size=opts.shuffle_size)
        ds = ds.batch(opts.batch_size)

        return ds
    return train_input_fn


def build_eval_input_fn(opts):
    def eval_input_fn():
        ds = create_image_dataset(opts.eval_data_path, has_label=True)
        num_parallel = opts.map_num_parallel_calls
        ds = ds.map(
            lambda filename, label: parse_function(filename, label, opts),
            num_parallel_calls=num_parallel)
        ds = ds.map(
            lambda image, label: val_preprocess(image, label, opts),
            num_parallel_calls=num_parallel)

        ds = ds.prefetch(opts.prefetch_size)
        if opts.multi_scale_predict and opts.inference_shape is None:
            ds = ds.batch(1)
        else:
            ds = ds.batch(opts.batch_size)
        return ds
    return eval_input_fn


def build_predict_input_fn(opts):
    def predict_input_fn():
        ds = create_image_dataset(opts.predict_data_path, has_label=False)
        num_parallel = opts.map_num_parallel_calls
        ds = ds.map(
            lambda filename: parse_function(filename, _DUMMY_LABEL, opts),
            num_parallel_calls=num_parallel)
        ds = ds.map(
            lambda filename: val_preprocess(filename, _DUMMY_LABEL, opts),
            num_parallel_calls=num_parallel)
        ds = ds.prefetch(opts.prefetch_size)
        if opts.multi_scale_predict:
            ds = ds.batch(1)
        else:
            ds = ds.batch(opts.batch_size)
        return ds
    return predict_input_fn


def build_serving_input_fn(opts):
    def serving_input_receiver_fn():
        feature_spec = {
            model_keys.DATA_COL: tf.FixedLenFeature(shape=INPUT_SHAPE,
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


###############################################################################

# easy version
# (1) decode image
def easy_parse_function(img_path, label, opts):

    image_string = tf.read_file(img_path)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    image_decoded.set_shape([None, None, 3])
    image = tf.cast(image_decoded, tf.float32)

    if label is _DUMMY_LABEL:
        return image
    else:
        return image, label


# (2) resize image to [224, 224, 3]
# (3) Substract the per color mean `VGG_MEAN`
# (4) RGB -> BGR
def easy_train_preprocess(image, label, opts):
    resized_image = tf.image.resize_images(image, INPUT_SHAPE[:-1])
    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = resized_image - means
    bgr_image = centered_image[:, :, ::-1]  # RGB -> BGR

    label = tf.one_hot(label, opts.num_classes)
    return {model_keys.DATA_COL: bgr_image}, label


# (2) resize image to [224, 224, 3]
# (3) Substract the per color mean `VGG_MEAN`
# (4) RGB -> BGR
def easy_val_preprocess(image, label, opts):
    if opts.multi_scale_predict:
        if opts.inference_shape is not None:
            resized_image = tf.image.resize_images(image, opts.inference_shape)
        else:
            resized_image = image  # do not resize image
    else:
        resized_image = tf.image.resize_images(image, INPUT_SHAPE[:-1])

    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = resized_image - means
    bgr_image = centered_image[:, :, ::-1]  # RGB -> BGR

    if label is not _DUMMY_LABEL:
        label = tf.one_hot(label, opts.num_classes)
        return {model_keys.DATA_COL: bgr_image}, label
    else:
        return {model_keys.DATA_COL: bgr_image}  # for predict


def build_easy_train_input_fn(opts):
    def easy_train_input_fn():
        ds = create_image_dataset(opts.train_data_path, has_label=True)
        num_parallel = opts.map_num_parallel_calls
        ds = ds.map(
            lambda filename, label: easy_parse_function(filename, label, opts),
            num_parallel_calls=num_parallel)
        ds = ds.map(
            lambda image, label: easy_train_preprocess(image, label, opts),
            num_parallel_calls=num_parallel)
        ds = ds.prefetch(opts.prefetch_size)
        if opts.shuffle_batch:
            ds = ds.shuffle(buffer_size=opts.shuffle_size)
        ds = ds.batch(opts.batch_size)

        return ds
    return easy_train_input_fn


def build_easy_eval_input_fn(opts):
    def easy_eval_input_fn():
        ds = create_image_dataset(opts.eval_data_path, has_label=True)
        num_parallel = opts.map_num_parallel_calls
        ds = ds.map(
            lambda filename, label: easy_parse_function(filename, label, opts),
            num_parallel_calls=num_parallel)
        ds = ds.map(
            lambda image, label: easy_val_preprocess(image, label, opts),
            num_parallel_calls=num_parallel)
        ds = ds.prefetch(opts.prefetch_size)
        if opts.multi_scale_predict and opts.inference_shape is None:
            ds = ds.batch(1)
        else:
            ds = ds.batch(opts.batch_size)
        return ds
    return easy_eval_input_fn


def build_easy_predict_input_fn(opts):
    def easy_predict_input_fn():
        ds = create_image_dataset(opts.predict_data_path, has_label=False)
        num_parallel = opts.map_num_parallel_calls
        ds = ds.map(
            lambda filename: easy_parse_function(filename, _DUMMY_LABEL, opts),
            num_parallel_calls=num_parallel)
        ds = ds.map(
            lambda filename: val_preprocess(filename, _DUMMY_LABEL, opts),
            num_parallel_calls=num_parallel)
        ds = ds.prefetch(opts.prefetch_size)
        if opts.multi_scale_predict:
            ds = ds.batch(1)
        else:
            ds = ds.batch(opts.batch_size)
        return ds
    return easy_predict_input_fn
