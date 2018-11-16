#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.preprocessing import normalize

import tensorflow as tf
import numpy as np
import os
import time

from preprocessing import preprocessing_factory
from nets import inception
from nets import resnet_v2
from nets import vgg

slim = tf.contrib.slim

# get script path
basedir = os.path.split(os.path.realpath(__file__))[0]
pretrained_checkpoints_path = os.path.join(basedir,
                                           '../pretrained_checkpoints')

tf.app.flags.DEFINE_string(
    'model_name', 'resnet_v2_50',
    'Specify the model used, one of "resnet_v2_50", "inception_v3"')

tf.app.flags.DEFINE_string(
    'preprocessing_name', 'inception',
    'Specify the image preprocessing methed, one of "inception", "vgg", '
    'resnet_v2_50 and inception_v3 suggest "inception".')

tf.app.flags.DEFINE_integer(
    'image_size', 299,
    'Preprocess image size, resnet_v2_50 and inception_v3 suggest 299')

tf.app.flags.DEFINE_string(
    'input', 'test.txt',
    'image path file, one per line')

tf.app.flags.DEFINE_string(
    'output', 'features.txt',
    'output image features file, one per line')

tf.app.flags.DEFINE_integer(
    'batch_size', 64,
    'batch size, as large as the machine can handle')

tf.app.flags.DEFINE_bool(
    'normalize', True,
    'weather normalize image vector.')

FLAGS = tf.app.flags.FLAGS


inception_v3_ckpt_path = os.path.join(pretrained_checkpoints_path,
                                      'inception_v3.ckpt')
resnet_v2_50_ckpt_path = os.path.join(pretrained_checkpoints_path,
                                      'resnet_v2_50.ckpt')
vgg_19_ckpt_path = os.path.join(pretrained_checkpoints_path,
                                'vgg_19.ckpt')


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
    image_decoded = image_decoded[:, :, :3]
    image_decoded.set_shape([None, None, 3])

    preprocessing_name = FLAGS.preprocessing_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)
    image = image_preprocessing_fn(image_decoded, image_size, image_size)

    return image, img_path


def eval_input_fn(data_path):
    ds = create_image_dataset(data_path)
    ds = ds.map(
        lambda filename: parse_function(filename, FLAGS.image_size),
        num_parallel_calls=1)
    ds = ds.batch(FLAGS.batch_size)

    return ds


def inception_v3(inputs):
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(
            inputs,
            num_classes=None,
            is_training=False,
            spatial_squeeze=True,
            global_pool=True)
    return logits, end_points, inception_v3_ckpt_path


def resnet_v2_50(inputs):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, end_points = resnet_v2.resnet_v2_50(
            inputs,
            num_classes=None,
            is_training=False,
            global_pool=True,
            spatial_squeeze=True)
    return logits, end_points, resnet_v2_50_ckpt_path


def vgg_19(inputs):
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_19(
            inputs,
            num_classes=None,
            is_training=False,
            fc_conv_padding='VALID',
            global_pool=True)
    return logits, end_points, vgg_19_ckpt_path


def get_model_def(model_name, inputs):
    model_def_map = {
        'inception_v3': inception_v3,
        'resnet_v2_50': resnet_v2_50,
        'vgg_19': vgg_19,
    }
    if model_name not in model_def_map:
        raise ValueError('Model name [%s] was not recognized' % model_name)

    model_def = model_def_map[model_name]
    return model_def(inputs)


def create_restore_fn(checkpoint_path):
    variables_to_restore = slim.get_variables_to_restore()

    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    tf.logging.info('checkpoint_path = {}'.format(checkpoint_path))
    restore_fn = slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=False)

    return restore_fn


def write_to_file(f, net, filenames):
    net = np.squeeze(net, (1, 2))
    if FLAGS.normalize:
        net = normalize(net, axis=1)

    for idx, features in enumerate(net):
        f.write(filenames[idx] + ' ')
        for value in features:
            f.write(str(value))
            f.write(' ')
        f.write('\n')


def display_flags():
    D = FLAGS.flag_values_dict()
    tf.logging.info("FLAGS: ")
    for key in D:
        tf.logging.info('{} = {}'.format(key, D[key]))


def main(_):
    display_flags()

    ds = eval_input_fn(FLAGS.input)
    iterator = ds.make_initializable_iterator()
    imgs_tensor, filenames_tensor = iterator.get_next()

    net, _, checkpoint_path = get_model_def(FLAGS.model_name, imgs_tensor)
    restore_fn = create_restore_fn(checkpoint_path)

    with tf.Session() as sess, open(FLAGS.output, 'w') as f:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(iterator.initializer)
        restore_fn(sess)

        total = 0
        while True:
            try:
                start_time = time.time()
                net_value, filenames = sess.run([net, filenames_tensor])
                write_to_file(f, net_value, filenames)
                end_time = time.time()
                duration_sec = end_time - start_time
                batch_size = len(net_value)
                total += batch_size
                tf.logging.info(
                    "{} images processed, cost {:.3f}s, {:.3f}ms per image, "
                    "total processed images {} ..."
                    .format(batch_size, duration_sec,
                            duration_sec*1000.0/batch_size, total))
            except tf.errors.OutOfRangeError:
                tf.logging.info("{} images processed, done.".format(total))
                break
            except Exception as e:
                tf.logging.info("{} run error, Exception = {}".format(e))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
