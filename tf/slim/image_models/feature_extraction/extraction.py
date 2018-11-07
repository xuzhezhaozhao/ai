#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.preprocessing import normalize

import tensorflow as tf
import numpy as np

from preprocessing import preprocessing_factory
from nets import inception

slim = tf.contrib.slim


tf.app.flags.DEFINE_string('input', 'input.txt', '')
tf.app.flags.DEFINE_string('output', 'output.txt', '')
tf.app.flags.DEFINE_integer('batch_size', 16, '')
tf.app.flags.DEFINE_bool('normalize', True, '')

FLAGS = tf.app.flags.FLAGS


def read_txt_file(txt_file):
    """Read the content of the text file and store it into lists."""

    img_paths = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            items = line.split(' ')
            img_paths.append(items[0])
    return img_paths


def create_image_dataset(data_path):
    img_paths = read_txt_file(data_path)
    img_paths = tf.convert_to_tensor(img_paths, dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices((img_paths))

    return ds


def parse_function(img_path, image_size):
    image_string = tf.read_file(img_path)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    image_decoded.set_shape([None, None, 3])

    preprocessing_name = 'inception'
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)
    image = image_preprocessing_fn(image_decoded, image_size, image_size)

    return image


def eval_input_fn(data_path):
    ds = create_image_dataset(data_path)
    ds = ds.map(
        lambda filename: parse_function(filename, 299), num_parallel_calls=1)
    ds = ds.batch(FLAGS.batch_size)

    return ds


def inception_v3(inputs):
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        return inception.inception_v3(
            inputs,
            num_classes=None,
            is_training=False,
            spatial_squeeze=True,
            global_pool=True)


def create_restore_fn():
    variables_to_restore = slim.get_variables_to_restore()
    tf.logging.info('Restore variables: ')
    for var in variables_to_restore:
        tf.logging.info(var)

    checkpoint_path = '../pretrained_checkpoint/inception_v3.ckpt'
    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    tf.logging.info('checkpoint_path = {}'.format(checkpoint_path))
    restore_fn = slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=False)

    tf.logging.info('Global trainable variables: ')
    for var in slim.get_trainable_variables():
        tf.logging.info(var)

    return restore_fn


def write_to_file(f, net):
    net = np.squeeze(net, (1, 2))
    net = normalize(net, axis=1)

    for features in net:
        for value in features:
            f.write(str(value))
            f.write(' ')
        f.write('\n')


def main(_):
    ds = eval_input_fn(FLAGS.input)
    iterator = ds.make_initializable_iterator()
    next_batch = iterator.get_next()
    net, _ = inception_v3(next_batch)
    restore_fn = create_restore_fn()

    with tf.Session() as sess, open(FLAGS.output, 'w') as f:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(iterator.initializer)
        restore_fn(sess)

        while True:
            try:
                net_value = sess.run(net)
                write_to_file(f, net_value)
            except tf.errors.OutOfRangeError:
                break


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run()
