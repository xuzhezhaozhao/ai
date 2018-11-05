#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets.nasnet import nasnet
from nets.nasnet import pnasnet
from nets import mobilenet_v1
from nets.mobilenet import mobilenet_v2
from nets import resnet_v1
from nets import resnet_v2
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
        'vgg_16': vgg_16,
        'vgg_19': vgg_19,
        'inception_v1': inception_v1,
        'inception_v2': inception_v2,
        'inception_v3': inception_v3,
        'inception_v4': inception_v4,
        'inception_resnet_v2': inception_resnet_v2,
        'resnet_v1_50': resnet_v1_50,
        'resnet_v1_101': resnet_v1_101,
        'resnet_v1_152': resnet_v1_152,
        'resnet_v2_50': resnet_v2_50,
        'resnet_v2_101': resnet_v2_101,
        'resnet_v2_152': resnet_v2_152,
        'mobilenet_v1_0.25_128': mobilenet_v1_025,
        'mobilenet_v1_0.5_160': mobilenet_v1_050,
        'mobilenet_v1_1.0_224': mobilenet_v1_100,
        'mobilenet_v2_1.0_224': mobilenet_v2_100,
        'mobilenet_v2_1.4_224': mobilenet_v2_140,
        'nasnet_mobile': nasnet_mobile,
        'nasnet_large': nasnet_large,
        'pnasnet_mobile': pnasnet_mobile,
        'pnasnet_large': pnasnet_large,
    }

    if model_name not in model_def_map:
        raise ValueError('Model name [%s] was not recognized' % model_name)

    model_def = model_def_map[model_name]
    return model_def(inputs, is_training, opts)


def vgg_16(inputs, is_training, opts):
    with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=opts.weight_decay)):
        return vgg.vgg_16(
            inputs,
            num_classes=opts.num_classes,
            dropout_keep_prob=opts.dropout_keep_prob,
            spatial_squeeze=opts.spatial_squeeze,
            is_training=is_training,
            fc_conv_padding='VALID',
            global_pool=opts.global_pool)


def vgg_19(inputs, is_training, opts):
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
            output_stride=None,
            spatial_squeeze=opts.spatial_squeeze,
            reuse=None)


def resnet_v1_101(inputs, is_training, opts):
    with slim.arg_scope(resnet_v1.resnet_arg_scope(
            weight_decay=opts.weight_decay,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon,
            activation_fn=tf.nn.relu)):
        return resnet_v1.resnet_v1_101(
            inputs,
            num_classes=opts.num_classes,
            is_training=is_training,
            global_pool=opts.global_pool,
            output_stride=None,
            spatial_squeeze=opts.spatial_squeeze,
            reuse=None)


def resnet_v1_152(inputs, is_training, opts):
    with slim.arg_scope(resnet_v1.resnet_arg_scope(
            weight_decay=opts.weight_decay,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon,
            activation_fn=tf.nn.relu)):
        return resnet_v1.resnet_v1_152(
            inputs,
            num_classes=opts.num_classes,
            is_training=is_training,
            global_pool=opts.global_pool,
            output_stride=None,
            spatial_squeeze=opts.spatial_squeeze,
            reuse=None)


def resnet_v2_50(inputs, is_training, opts):
    with slim.arg_scope(resnet_v2.resnet_arg_scope(
            weight_decay=opts.weight_decay,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon,
            activation_fn=tf.nn.relu)):
        return resnet_v2.resnet_v2_50(
            inputs,
            num_classes=opts.num_classes,
            is_training=is_training,
            global_pool=opts.global_pool,
            output_stride=None,
            spatial_squeeze=opts.spatial_squeeze,
            reuse=None)


def resnet_v2_101(inputs, is_training, opts):
    with slim.arg_scope(resnet_v2.resnet_arg_scope(
            weight_decay=opts.weight_decay,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon,
            activation_fn=tf.nn.relu)):
        return resnet_v2.resnet_v2_101(
            inputs,
            num_classes=opts.num_classes,
            is_training=is_training,
            global_pool=opts.global_pool,
            output_stride=None,
            spatial_squeeze=opts.spatial_squeeze,
            reuse=None)


def resnet_v2_152(inputs, is_training, opts):
    with slim.arg_scope(resnet_v2.resnet_arg_scope(
            weight_decay=opts.weight_decay,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon,
            activation_fn=tf.nn.relu)):
        return resnet_v2.resnet_v2_152(
            inputs,
            num_classes=opts.num_classes,
            is_training=is_training,
            global_pool=opts.global_pool,
            output_stride=None,
            spatial_squeeze=opts.spatial_squeeze,
            reuse=None)


# see https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md
def mobilenet_v1_025(inputs, is_training, opts):
    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(
            is_training=is_training,
            weight_decay=opts.weight_decay,
            stddev=0.09,
            regularize_depthwise=False,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon)):
        return mobilenet_v1.mobilenet_v1_025(
            inputs,
            num_classes=opts.num_classes,
            dropout_keep_prob=opts.dropout_keep_prob,
            is_training=is_training,
            min_depth=8,
            global_pool=opts.global_pool,
            spatial_squeeze=opts.spatial_squeeze,
            reuse=None)


def mobilenet_v1_050(inputs, is_training, opts):
    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(
            is_training=is_training,
            weight_decay=opts.weight_decay,
            stddev=0.09,
            regularize_depthwise=False,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon)):
        return mobilenet_v1.mobilenet_v1_050(
            inputs,
            num_classes=opts.num_classes,
            dropout_keep_prob=opts.dropout_keep_prob,
            is_training=is_training,
            min_depth=8,
            global_pool=opts.global_pool,
            spatial_squeeze=opts.spatial_squeeze,
            reuse=None)


def mobilenet_v1_100(inputs, is_training, opts):
    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(
            is_training=is_training,
            weight_decay=opts.weight_decay,
            stddev=0.09,
            regularize_depthwise=False,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon)):
        return mobilenet_v1.mobilenet_v1(
            inputs,
            num_classes=opts.num_classes,
            dropout_keep_prob=opts.dropout_keep_prob,
            is_training=is_training,
            min_depth=8,
            depth_multiplier=1.0,
            global_pool=opts.global_pool,
            spatial_squeeze=opts.spatial_squeeze,
            reuse=None)


def mobilenet_v2_100(inputs, is_training, opts):
    if is_training:
        with slim.arg_scope(mobilenet_v2.training_scope(
                weight_decay=opts.weight_decay,
                stddev=0.09,
                bn_decay=opts.batch_norm_decay)):
            return mobilenet_v2.mobilenet(
                inputs,
                num_classes=opts.num_classes,
                depth_multiplier=1.0,
                reuse=None)
    else:
        return mobilenet_v2.mobilenet(
            inputs,
            num_classes=opts.num_classes,
            depth_multiplier=1.0,
            reuse=None)


def mobilenet_v2_140(inputs, is_training, opts):
    if is_training:
        with slim.arg_scope(mobilenet_v2.training_scope(
                weight_decay=opts.weight_decay,
                stddev=0.09,
                bn_decay=opts.batch_norm_decay)):
            return mobilenet_v2.mobilenet_v2_140(
                inputs,
                num_classes=opts.num_classes,
                reuse=None)
    else:
        return mobilenet_v2.mobilenet_v2_140(
            inputs,
            num_classes=opts.num_classes,
            reuse=None)


def nasnet_mobile(inputs, is_training, opts):
    with slim.arg_scope(nasnet.nasnet_mobile_arg_scope(
            weight_decay=opts.weight_decay,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon)):

        config = nasnet.mobile_imagenet_config()
        config.set_hparam('dense_dropout_keep_prob', opts.dropout_keep_prob)
        config.set_hparam('use_aux_head', int(opts.create_aux_logits))

        return nasnet.build_nasnet_mobile(
            inputs,
            num_classes=opts.num_classes,
            is_training=is_training,
            config=config)


def nasnet_large(inputs, is_training, opts):
    with slim.arg_scope(nasnet.nasnet_large_arg_scope(
            weight_decay=opts.weight_decay,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon)):

        config = nasnet.large_imagenet_config()
        config.set_hparam('dense_dropout_keep_prob', opts.dropout_keep_prob)
        config.set_hparam('use_aux_head', int(opts.create_aux_logits))

        return nasnet.build_nasnet_large(
            inputs,
            num_classes=opts.num_classes,
            is_training=is_training,
            config=config)


def pnasnet_mobile(inputs, is_training, opts):
    with slim.arg_scope(pnasnet.pnasnet_mobile_arg_scope(
            weight_decay=opts.weight_decay,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon)):

        config = pnasnet.large_imagenet_config()
        config.set_hparam('dense_dropout_keep_prob', opts.dropout_keep_prob)
        config.set_hparam('use_aux_head', int(opts.create_aux_logits))

        return pnasnet.build_pnasnet_mobile(
            inputs,
            num_classes=opts.num_classes,
            is_training=is_training,
            config=config)


def pnasnet_large(inputs, is_training, opts):
    with slim.arg_scope(pnasnet.pnasnet_large_arg_scope(
            weight_decay=opts.weight_decay,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon)):

        config = pnasnet.large_imagenet_config()
        config.set_hparam('dense_dropout_keep_prob', opts.dropout_keep_prob)
        config.set_hparam('use_aux_head', int(opts.create_aux_logits))

        return pnasnet.build_pnasnet_large(
            inputs,
            num_classes=opts.num_classes,
            is_training=is_training,
            config=config)
