#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets import vgg
from common import model_keys
from common import build_model_fn_utils


# build vgg16
def model_fn(features, labels, mode, params):
    """Build model graph."""

    tf.summary.image('images', features[model_keys.DATA_COL])

    opts = params['opts']
    inputs = features[model_keys.DATA_COL]
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=opts.l2_regularizer)):
        logits, end_points = vgg.vgg_16(
            inputs,
            num_classes=opts.num_classes,
            dropout_keep_prob=opts.dropout_keep_prob,
            is_training=is_training,
            fc_conv_padding='VALID',
            global_pool=opts.global_pool)

    return build_model_fn_utils.create_estimator_spec(
        mode, logits, labels, params)
