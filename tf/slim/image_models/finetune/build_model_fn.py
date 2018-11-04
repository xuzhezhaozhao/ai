#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets import resnet_v1
from nets import inception
from nets import vgg
import model_keys
import build_model_fn_utils


def model_fn(features, labels, mode, params):
    """Build model graph."""

    tf.summary.image('images', features[model_keys.DATA_COL])

    opts = params['opts']
    inputs = features[model_keys.DATA_COL]
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    logits, end_points = get_model_def(
        opts.model_name, inputs, is_training, opts)

    if opts.use_moving_average:
        variables_to_train = \
            build_model_fn_utils.get_finetune_trainable_variables(opts)
        ema = tf.train.ExponentialMovingAverage(opts.moving_average_decay)
        maintain_averages_op = ema.apply(variables_to_train)
    else:
        ema, maintain_averages_op = None, None

    return build_model_fn_utils.create_estimator_spec(
        mode, logits, labels, params, ema, maintain_averages_op)


def get_model_def(model_name, inputs, is_training, opts):
    model_def_map = {
        'vgg16': vgg16,
        'vgg19': vgg19,
        'inception_v1': inception_v1,
        'inception_v2': inception_v2,
        'inception_v3': inception_v3,
        'inception_v4': inception_v4,
        'inception_resnet_v2': inception_resnet_v2,
        'resnet_v1_50': resnet_v1_50,
    }

    if model_name not in model_def_map:
        raise ValueError('Model name [%s] was not recognized' % model_name)

    model_def = model_def_map[model_name]
    return model_def(inputs, is_training, opts)


def vgg16(inputs, is_training, opts):
    with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=opts.weight_decay)):
        return vgg.vgg_16(
            inputs,
            num_classes=opts.num_classes,
            dropout_keep_prob=opts.dropout_keep_prob,
            spatial_squeeze=opts.spatial_squeeze,
            is_training=is_training,
            fc_conv_padding='VALID',
            global_pool=opts.global_pool)


def vgg19(inputs, is_training, opts):
    with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=opts.weight_decay)):
        return vgg.vgg_19(
            inputs,
            num_classes=opts.num_classes,
            dropout_keep_prob=opts.dropout_keep_prob,
            spatial_squeeze=opts.spatial_squeeze,
            is_training=is_training,
            fc_conv_padding='VALID',
            global_pool=opts.global_pool)


def inception_v1(inputs, is_training, opts):
    with slim.arg_scope(inception.inception_v1_arg_scope(
            weight_decay=opts.weight_decay,
            use_batch_norm=opts.use_batch_norm,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon,
            activation_fn=tf.nn.relu)):
        return inception.inception_v1(
            inputs,
            num_classes=opts.num_classes,
            is_training=is_training,
            dropout_keep_prob=opts.dropout_keep_prob,
            prediction_fn=slim.softmax,
            spatial_squeeze=opts.spatial_squeeze,
            reuse=None,
            global_pool=opts.global_pool)


def inception_v2(inputs, is_training, opts):
    with slim.arg_scope(inception.inception_v2_arg_scope(
            weight_decay=opts.weight_decay,
            use_batch_norm=opts.use_batch_norm,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon,
            activation_fn=tf.nn.relu)):
        return inception.inception_v2(
            inputs,
            num_classes=opts.num_classes,
            is_training=is_training,
            dropout_keep_prob=opts.dropout_keep_prob,
            min_depth=opts.min_depth,
            depth_multiplier=opts.depth_multiplier,
            prediction_fn=slim.softmax,
            spatial_squeeze=opts.spatial_squeeze,
            reuse=None,
            global_pool=opts.global_pool)


def inception_v3(inputs, is_training, opts):
    with slim.arg_scope(inception.inception_v3_arg_scope(
            weight_decay=opts.weight_decay,
            use_batch_norm=opts.use_batch_norm,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon,
            activation_fn=tf.nn.relu)):
        return inception.inception_v3(
            inputs,
            num_classes=opts.num_classes,
            is_training=is_training,
            dropout_keep_prob=opts.dropout_keep_prob,
            min_depth=opts.min_depth,
            depth_multiplier=opts.depth_multiplier,
            prediction_fn=slim.softmax,
            spatial_squeeze=opts.spatial_squeeze,
            reuse=None,
            create_aux_logits=opts.create_aux_logits,
            global_pool=opts.global_pool)


def inception_v4(inputs, is_training, opts):
    with slim.arg_scope(inception.inception_v4_arg_scope(
            weight_decay=opts.weight_decay,
            use_batch_norm=opts.use_batch_norm,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon,
            activation_fn=tf.nn.relu)):
        return inception.inception_v4(
            inputs,
            num_classes=opts.num_classes,
            is_training=is_training,
            dropout_keep_prob=opts.dropout_keep_prob,
            reuse=None,
            create_aux_logits=opts.create_aux_logits)


def inception_resnet_v2(inputs, is_training, opts):
    with slim.arg_scope(inception.inception_resnet_v2_arg_scope(
            weight_decay=opts.weight_decay,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon,
            activation_fn=tf.nn.relu)):
        return inception.inception_resnet_v2(
            inputs,
            num_classes=opts.num_classes,
            is_training=is_training,
            dropout_keep_prob=opts.dropout_keep_prob,
            reuse=None,
            create_aux_logits=opts.create_aux_logits)


def resnet_v1_50(inputs, is_training, opts):
    with slim.arg_scope(resnet_v1.resnet_arg_scope(
            weight_decay=opts.weight_decay,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon,
            activation_fn=tf.nn.relu)):
        return resnet_v1.resnet_v1_50(
            inputs,
            num_classes=opts.num_classes,
            is_training=is_training,
            global_pool=opts.global_pool,
            reuse=None)
