#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.misc
import tensorflow as tf
import numpy as np
import vgg19

from transform_net import transform_net


def model_fn(features, labels, mode, params):
    """Build model graph."""

    inputs = features['data']
    tf.summary.image("inputs", inputs)
    opts = params['opts']

    style_grams = preprocess_style(opts)

    # compute target content features
    vgg_target = vgg19.Vgg19(opts.vgg19_npy_path)
    vgg_target.build(inputs, sub_mean=False, name='vgg_target')

    # forward transform net and compute predict content features
    transform_image = transform_net('transform_net', inputs)
    tf.summary.image("transform_image", tf.cast(tf.clip_by_value(
        transform_image, 0, 255), tf.uint8))
    vgg_predict = vgg19.Vgg19(opts.vgg19_npy_path)
    vgg_predict.build(transform_image, sub_mean=True, name='vgg_predict')

    with tf.name_scope('content_loss'):
        content_loss = 0.0
        layer_weights = map(float, opts.content_layer_loss_weights)
        if len(layer_weights) == 1:
            layer_weights = layer_weights * len(opts.content_layers)
        elif len(layer_weights) != len(opts.content_layers):
            raise ValueError("content_layer_loss_weights not match "
                             "content_layers.")
        for layer, weight in zip(opts.content_layers, layer_weights):
            content_loss += weight * tf.losses.mean_squared_error(
                vgg_target.end_points[layer],
                vgg_predict.end_points[layer]) / 2.0
        tf.summary.scalar('content_loss', content_loss)

    with tf.name_scope('style_loss'):
        style_loss = 0.0
        layer_weights = map(float, opts.style_layer_loss_weights)
        if len(layer_weights) == 1:
            layer_weights = layer_weights * len(opts.style_layers)
        elif len(layer_weights) != len(opts.style_layers):
            raise ValueError("style_layer_loss_weights not match "
                             "style_layers.")
        for layer, weight in zip(opts.style_layers, layer_weights):
            feature_map = vgg_predict.end_points[layer]
            _, h, w, filters = map(lambda s: s.value,
                                   feature_map.get_shape())
            feature_map = tf.reshape(feature_map, (-1, h*w, filters))
            gram = tf.matmul(tf.transpose(feature_map, perm=[0, 2, 1]),
                             feature_map)
            gram /= float(h*w*filters)
            style_loss += weight * tf.losses.mean_squared_error(
                style_grams[layer], gram) / 4.0
        tf.summary.scalar('style_loss', style_loss)

    with tf.name_scope('loss'):
        loss = opts.content_loss_weight * content_loss \
            + opts.style_loss_weight * style_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        learning_rate = configure_learning_rate(global_step, opts)
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = configure_optimizer(learning_rate, opts)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.contrib.training.create_train_op(
            total_loss=loss,
            optimizer=optimizer,
            global_step=global_step,
            update_ops=update_ops,
            variables_to_train=tf.trainable_variables(),
            transform_grads_fn=None,
            summarize_gradients=True,
            aggregation_method=None,
            colocate_gradients_with_ops=False,
            check_numerics=True)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={})


def configure_learning_rate(global_step, opts):
    decay_steps = opts.decay_steps

    with tf.variable_scope("configure_learning_rate"):
        if opts.learning_rate_decay_type == 'exponential':
            return tf.train.exponential_decay(
                opts.learning_rate,
                global_step,
                decay_steps,
                opts.learning_rate_decay_factor,
                staircase=True,
                name='exponential_decay_learning_rate')
        elif opts.learning_rate_decay_type == 'fixed':
            return tf.constant(opts.learning_rate, name='fixed_learning_rate')
        elif opts.learning_rate_decay_type == 'polynomial':
            return tf.train.polynomial_decay(
                opts.learning_rate,
                global_step,
                decay_steps,
                opts.end_learning_rate,
                power=1.0,
                cycle=False,
                name='polynomial_decay_learning_rate')
        else:
            raise ValueError('learning_rate_decay_type [%s] was not recognized'
                             % opts.learning_rate_decay_type)


def configure_optimizer(learning_rate, opts):
    """Configures the optimizer used for training."""

    with tf.variable_scope("configure_optimizer"):
        if opts.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(
                learning_rate,
                rho=opts.adadelta_rho,
                epsilon=opts.opt_epsilon)
        elif opts.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
                learning_rate,
                initial_accumulator_value=opts.adagrad_init_value)
        elif opts.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate,
                beta1=opts.adam_beta1,
                beta2=opts.adam_beta2,
                epsilon=opts.opt_epsilon)
        elif opts.optimizer == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(
                learning_rate,
                learning_rate_power=opts.ftrl_learning_rate_power,
                initial_accumulator_value=opts.ftrl_initial_accumulator_value,
                l1_regularization_strength=opts.ftrl_l1,
                l2_regularization_strength=opts.ftrl_l2)
        elif opts.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate,
                momentum=opts.momentum,
                name='Momentum')
        elif opts.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate,
                decay=opts.rmsprop_decay,
                momentum=opts.rmsprop_momentum,
                epsilon=opts.opt_epsilon)
        elif opts.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('Optimizer [%s] was not recognized'
                             % opts.optimizer)
        return optimizer


def preprocess_style(opts):
    style_grams = {}
    with tf.Graph().as_default():
        vgg = vgg19.Vgg19(opts.vgg19_npy_path)
        image = imread(opts.style_image_path)
        image = np.expand_dims(image, 0)
        vgg.build(image, sub_mean=True)
        with tf.Session() as sess:
            for layer in opts.style_layers:
                feature_map = sess.run(vgg.end_points[layer])
                feature_map = np.reshape(feature_map,
                                         (-1, feature_map.shape[3]))
                gram = np.matmul(feature_map.T, feature_map)
                gram /= feature_map.size
                gram = np.expand_dims(gram, 0)
                style_grams[layer] = gram
                tf.logging.info("layer {} gram matrix shape: {}"
                                .format(layer, gram.shape))
    return style_grams


def imread(img_path):
    img = scipy.misc.imread(img_path).astype(np.float32)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:, :, :3]
    return img
