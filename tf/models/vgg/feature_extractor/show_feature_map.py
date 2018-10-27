#! /usr/bin/env python
# -*- coding=utf8 -*-

"""
ref:
https://blog.csdn.net/c20081052/article/details/80207906?utm_source=blogxgwz9

imshow: https://blog.csdn.net/Goldxwang/article/details/76855200
"""


import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

import vgg19


parser = argparse.ArgumentParser()
parser.add_argument('--img_path', default='', type=str, help='')
parser.add_argument('--subplot', default=0, type=int, help='')
parser.add_argument('--feature_map', default=0, type=int, help='')


INPUT_SHAPE = [224, 224]
VGG_MEAN = [103.939, 116.779, 123.68]
LAYERS = (
    ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1'],
    ['conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2'],
    ['conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
     'relu3_3', 'conv3_4', 'relu3_4', 'pool3'],
    ['conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
     'relu4_3', 'conv4_4', 'relu4_4', 'pool4'],
    ['conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
     'relu5_3', 'conv5_4', 'relu5_4', 'pool5']
)


def parse_img(img_path):
    # load and preprocess the image
    img_string = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_string, channels=3)
    img_decoded.set_shape([None, None, 3])

    img_centered = tf.subtract(tf.cast(img_decoded, tf.float32), VGG_MEAN)
    img_resized = tf.image.resize_images(img_centered, INPUT_SHAPE)

    # RGB -> BGR
    img_bgr = img_resized[:, :, ::-1]
    img_bgr = tf.expand_dims(img_bgr, 0)

    return img_bgr


def run(args):
    data = parse_img(args.img_path)
    vgg = vgg19.Vgg19()
    vgg.build(data)

    with tf.Session() as sess:
        if args.subplot:
            hspace = 0.6
            plt.subplots_adjust(hspace=hspace)
            for layer_index, layer in enumerate(LAYERS):
                for sublayer_index, sublayer in enumerate(layer):
                    features = sess.run(vgg.__dict__[sublayer])
                    print("layer '{}', shape '{}'"
                          .format(sublayer, str(features.shape)))
                    plt.subplot(5, 9, (layer_index*9 + sublayer_index)+1)
                    # cmap: hot, spring, cool, bone
                    # or use plt.matshow
                    im = plt.imshow(features[0, :, :, args.feature_map],
                                    cmap=plt.cm.gray)
                    plt.title(sublayer, y=1.0)
                    plt.colorbar(im)
            plt.show()
        else:
            for index, layer in enumerate(LAYERS):
                features = sess.run(vgg.__dict__[layer])
                print("layer '{}', shape '{}'"
                      .format(layer, str(features.shape)))
                plt.figure(index+1)
                plt.matshow(features[0, :, :, 0], cmap=plt.cm.gray,
                            fignum=index+1)
                plt.title(layer)
                plt.colorbar()
                plt.show()


def main(argv):
    args = parser.parse_args(argv[1:])
    run(args)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
