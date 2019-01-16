#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import vgg19


def model_fn(features, labels, mode, params):
    """Build model graph."""

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    inputs = features['data']
    tf.logging.info("image size: {}".format(inputs))
    opts = params['opts']

    vgg_predict = vgg19.Vgg19(opts.vgg19_npy_path)

    # TODO pre-compute style grams
    style_grams = {}

    # compute content features
    vgg_target = vgg19.Vgg19(opts.vgg19_npy_path)
    vgg_target.build(inputs)
    # forward transform net and compute predict content features
    transform_image = None  # TODO
    vgg_target = vgg19.Vgg19(opts.vgg19_npy_path)
    vgg_predict.build(transform_image)

    # content loss
    content_loss = 0.0
    layer_weights = map(float, opts.content_layer_loss_weights)
    if len(layer_weights) == 1:
        layer_weights = layer_weights * len(opts.content_layers)
    elif len(layer_weights) != len(opts.content_layers):
        raise ValueError("content_layer_loss_weights not match "
                         "content_layers.")
    for layer, weight in zip(opts.content_layers, layer_weights):
        content_loss += weight * tf.losses.mean_squared_error(
            vgg_target.end_points[layer], vgg_predict.end_points[layer]) / 2.0
    tf.summary.scalar('content_loss', content_loss)

    # style loss
    style_loss = 0.0
    layer_weights = map(float, opts.style_layer_loss_weights)
    if len(layer_weights) == 1:
        layer_weights = layer_weights * len(opts.style_layers)
    elif len(layer_weights) != len(opts.style_layers):
        raise ValueError("style_layer_loss_weights not match "
                         "style_layers.")
    for layer, weight in zip(opts.style_layers, layer_weights):
        feature_map = vgg_predict.end_points[layer]
        feature_map = tf.reshape(feature_map, (-1, feature_map.shape[3]))
        gram = tf.matmul(tf.transpose(feature_map), feature_map)
        gram /= tf.cast(tf.size(feature_map), tf.float32)
        style_loss += weight * tf.losses.mean_squared_error(
            style_grams[layer], gram) / 4.0
    tf.summary.scalar('style_loss', style_loss)

    loss = opts.content_loss_weight * content_loss \
        + opts.style_loss_weight * style_loss
    tf.summary.scalar('loss', loss)
