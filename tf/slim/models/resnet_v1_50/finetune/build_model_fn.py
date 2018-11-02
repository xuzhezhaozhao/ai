#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets import resnet_v1
from common import model_keys
from common import build_model_fn_utils


# build resnet_v1_50
def model_fn(features, labels, mode, params):
    """Build model graph."""

    tf.summary.image('images', features[model_keys.DATA_COL])

    opts = params['opts']
    inputs = features[model_keys.DATA_COL]
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(
            weight_decay=opts.l2_regularizer,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon,
            activation_fn=tf.nn.relu)):
        logits, end_points = resnet_v1.resnet_v1_50(
            inputs,
            num_classes=opts.num_classes,
            is_training=is_training,
            global_pool=opts.global_pool,
            reuse=None)

    return build_model_fn_utils.create_estimator_spec(
        mode, logits, labels, params)
