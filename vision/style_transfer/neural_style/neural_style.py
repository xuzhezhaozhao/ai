#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import reduce
from operator import mul

import tensorflow as tf
import numpy as np
import vgg19


CONTENT_LAYERS = ('conv4_2', 'conv5_2')
STYLE_LAYERS = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1')


class NeuralStyle(object):
    def __init__(self, vgg19_npy_path, content_image, style_image, init_image):
        self.vgg19_npy_path = vgg19_npy_path
        self.content_image = content_image
        self.style_image = style_image
        self.init_image = init_image

        self.preprocess()
        self.build_graph()

    def preprocess(self):
        self.content_features = {}
        self.style_grams = {}

        vgg = vgg19.Vgg19(self.vgg19_npy_path)
        content_image = np.expand_dims(self.content_image, 0)
        style_image = np.expand_dims(self.style_image, 0)
        image = tf.placeholder(tf.float32, shape=(1, None, None, 3))
        vgg.build(image)
        with tf.Session() as sess:
            for layer in CONTENT_LAYERS:
                feature_map = sess.run(
                    vgg.end_points[layer], feed_dict={image: content_image})
                self.content_features[layer] = feature_map
                print("layer {} feature map shape: {}"
                      .format(layer, feature_map.shape))
            for layer in STYLE_LAYERS:
                feature_map = sess.run(vgg.end_points[layer],
                                       feed_dict={image: content_image})
                feature_map = np.reshape(feature_map,
                                         (-1, feature_map.shape[3]))
                gram = np.matmul(feature_map.T, feature_map)
                gram /= feature_map.size
                self.style_grams[layer] = gram
                print("layer {} gram matrix shape: {}"
                      .format(layer, gram.shape))

    def build_graph(self):
        init_image = np.expand_dims(self.init_image, 0)
        vgg = vgg19.Vgg19(self.vgg19_npy_path)
        self.output_image = tf.get_variable(
            "output_image", initializer=init_image)
        vgg.build(self.output_image)
        # content loss
        self.content_loss = 0.0
        for layer in CONTENT_LAYERS:
            self.content_loss += tf.losses.mean_squared_error(
                self.content_features[layer], vgg.end_points[layer])

        # style loss
        self.style_loss = 0.0
        for layer in STYLE_LAYERS:
            feature_map = vgg.end_points[layer]
            feature_map = tf.reshape(feature_map,
                                     (-1, feature_map.shape[3]))
            gram = tf.matmul(tf.transpose(feature_map), feature_map)
            nelems = reduce(mul, feature_map.shape.as_list())
            gram /= nelems
            self.style_loss += tf.losses.mean_squared_error(
                self.style_grams[layer], gram)

        self.loss = self.content_loss + self.style_loss
