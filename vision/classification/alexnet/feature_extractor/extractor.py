#! /usr/bin/env python
# -*- coding=utf8 -*-


import tensorflow as tf
import argparse

import alexnet


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='', type=str, help='')
parser.add_argument('--batch_size', default=256, type=int, help='')
parser.add_argument('--feature_layer', default='', type=str, help='')


INPUT_SHAPE = [227, 227]
# RGB
IMAGENET_MEAN = [123.68, 116.779, 103.939]


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


def extractor_input_fn(txt_file, args):
    img_paths = read_txt_file(txt_file)
    img_paths = tf.convert_to_tensor(img_paths, dtype=tf.string)

    # create dataset
    ds = tf.data.Dataset.from_tensor_slices((img_paths))
    ds = ds.map(lambda filename: parse_line(filename),
                num_parallel_calls=1)

    ds = ds.prefetch(args.batch_size)
    ds = ds.batch(args.batch_size)
    return ds


def parse_line(img_path):
    # load and preprocess the image
    img_string = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_string, channels=3)
    img_decoded.set_shape([None, None, 3])

    img_centered = tf.subtract(tf.cast(img_decoded, tf.float32), IMAGENET_MEAN)
    img_resized = tf.image.resize_images(img_centered, INPUT_SHAPE)

    # RGB -> BGR
    img_bgr = img_resized[:, :, ::-1]

    return {'data': img_bgr}


def run(args):
    txt_file = args.data_path
    ds = extractor_input_fn(txt_file, args)
    it = ds.make_initializable_iterator()
    data = it.get_next()['data']
    alex = alexnet.AlexNet()
    alex.build(data)

    with tf.Session() as sess:
        sess.run(it.initializer)
        while True:
            try:
                features = sess.run(alex.__dict__[args.feature_layer])
                print("layer '{}', shape '{}'"
                      .format(args.feature_layer, features.shape))
            except tf.errors.OutOfRangeError:
                break


def main(argv):
    args = parser.parse_args(argv[1:])
    run(args)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
