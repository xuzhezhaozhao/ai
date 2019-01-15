#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.misc
import tensorflow as tf
import numpy as np
import vgg19


class NeuralStyle(object):
    def __init__(self, opts):
        self.opts = opts
        style_image = imread(opts.style_image_path)
        content_image = imread(opts.content_image_path)
        init_image = None
        if opts.use_init_image:
            init_image = imread(opts.init_image_path)
            init_image = scipy.misc.imresize(
                init_image, content_image.shape[:2]).astype(np.float32)
        else:
            shape = content_image.shape
            init_image = np.random.randint(0, 255, shape).astype(np.float32)

        self.content_image = np.expand_dims(content_image, 0)
        self.style_image = np.expand_dims(style_image, 0)
        self.init_image = np.expand_dims(init_image, 0)
        self.preprocess()
        self.build_graph()

    def preprocess(self):
        """Calulate content feature maps and style gram matrixes."""

        self.content_features = {}
        self.style_grams = {}
        with tf.Graph().as_default():
            vgg = vgg19.Vgg19(self.opts.vgg19_npy_path)
            image = tf.placeholder(tf.float32, shape=(1, None, None, 3))
            vgg.build(image)
            with tf.Session() as sess:
                for layer in self.opts.content_layers:
                    feature_map = sess.run(
                        vgg.end_points[layer],
                        feed_dict={image: self.content_image})
                    self.content_features[layer] = feature_map
                    print("layer {} feature map shape: {}"
                          .format(layer, feature_map.shape))
                for layer in self.opts.style_layers:
                    feature_map = sess.run(vgg.end_points[layer],
                                           feed_dict={image: self.style_image})
                    feature_map = np.reshape(feature_map,
                                             (-1, feature_map.shape[3]))
                    gram = np.matmul(feature_map.T, feature_map)
                    gram /= feature_map.size
                    self.style_grams[layer] = gram
                    print("layer {} gram matrix shape: {}"
                          .format(layer, gram.shape))

    def build_graph(self):
        vgg = vgg19.Vgg19(self.opts.vgg19_npy_path)
        self.output_image = tf.get_variable(
            "output_image", initializer=self.init_image)
        vgg.build(self.output_image)

        # content loss
        self.content_loss = 0.0
        layer_weights = map(float, self.opts.content_layer_loss_weights)
        if len(layer_weights) == 1:
            layer_weights = layer_weights * len(self.opts.content_layers)
        elif len(layer_weights) != len(self.opts.content_layers):
            raise ValueError("content_layer_loss_weights not match "
                             "content_layers.")

        for layer, weight in zip(self.opts.content_layers, layer_weights):
            self.content_loss += weight * tf.losses.mean_squared_error(
                self.content_features[layer], vgg.end_points[layer]) / 2.0
        tf.summary.scalar('content_loss', self.content_loss)

        # style loss
        self.style_loss = 0.0
        layer_weights = map(float, self.opts.style_layer_loss_weights)
        if len(layer_weights) == 1:
            layer_weights = layer_weights * len(self.opts.style_layers)
        elif len(layer_weights) != len(self.opts.style_layers):
            raise ValueError("style_layer_loss_weights not match "
                             "style_layers.")
        for layer, weight in zip(self.opts.style_layers, layer_weights):
            feature_map = vgg.end_points[layer]
            feature_map = tf.reshape(feature_map,
                                     (-1, feature_map.shape[3]))
            gram = tf.matmul(tf.transpose(feature_map), feature_map)
            gram /= tf.cast(tf.size(feature_map), tf.float32)
            self.style_loss += weight * tf.losses.mean_squared_error(
                self.style_grams[layer], gram) / 4.0
        tf.summary.scalar('style_loss', self.style_loss)

        self.loss = self.opts.content_loss_weight * self.content_loss \
            + self.opts.style_loss_weight * self.style_loss
        tf.summary.scalar('loss', self.loss)


def imread(img_path):
    img = scipy.misc.imread(img_path).astype(np.float32)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:, :, :3]
    return img
