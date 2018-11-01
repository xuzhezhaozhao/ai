#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim.nets as nets

import model_keys


slim = tf.contrib.slim
vgg = nets.vgg

_global_learning_rate = 0.01


def set_global_learning_rate(rate):
    global _global_learning_rate
    _global_learning_rate = rate


def get_global_learning_rate():
    global _global_learning_rate
    return _global_learning_rate


# build vgg16
def vgg16_model_fn(features, labels, mode, params):
    """Build model graph."""

    tf.summary.image('images', features[model_keys.DATA_COL])

    opts = params['opts']
    inputs = features[model_keys.DATA_COL]

    training = (mode == tf.estimator.ModeKeys.TRAIN)

    fc8, endpoints = vgg.vgg_16(
        inputs, is_training=training, num_classes=opts.num_classes)

    score = tf.nn.softmax(fc8)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return create_predict_estimator_spec(mode, score, labels, params)

    if mode == tf.estimator.ModeKeys.EVAL:
        return create_eval_estimator_spec(mode, fc8, labels, params)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return create_train_estimator_spec(mode, fc8, labels, params)


def get_loss(logits, labels, name):
    with tf.name_scope(name):
        l2_loss = tf.losses.get_regularization_loss()
        ce_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=tf.argmax(labels, 1)))
        loss = l2_loss + ce_loss
        tf.summary.scalar("l2_loss", l2_loss)
        tf.summary.scalar("ce_loss", ce_loss)
        tf.summary.scalar("total_loss", loss)
    return loss


def create_predict_estimator_spec(mode, score, labels, params):
    """Create predict EstimatorSpec."""

    predictions = {
        'score': score,
    }

    export_outputs = {
        'predicts': tf.estimator.export.PredictOutput(
            outputs={
                'score': score,
            }
        )
    }

    return tf.estimator.EstimatorSpec(mode, predictions=predictions,
                                      export_outputs=export_outputs)


def create_eval_estimator_spec(mode, logits, labels, params):
    """Create eval EstimatorSpec."""

    accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                   predictions=tf.argmax(logits, 1))
    metrics = {
        'accuracy': accuracy
    }
    add_metrics_summary(metrics)

    loss = get_loss(logits, labels, 'eval')
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


def create_train_estimator_spec(mode, logits, labels, params):
    """Create train EstimatorSpec."""

    tf.summary.scalar('train_batch_accuracy', tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1)),
                tf.float32)))

    opts = params['opts']
    global_step = tf.train.get_global_step()
    loss = get_loss(logits, labels, 'train')
    lr = get_global_learning_rate()
    tf.logging.info("global learning rate = {}".format(lr))
    tf.summary.scalar('learning_rate', lr)
    optimizer = tf.train.MomentumOptimizer(
        lr, opts.optimizer_momentum_momentum)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    train_op = optimizer.apply_gradients(
        zip(gradients, variables), global_step=global_step)

    for var, grad in zip(variables, gradients):
        tf.summary.histogram(var.name.replace(':', '_') + '/gradient', grad)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def add_metrics_summary(metrics):
    """Add metrics to tensorboard."""

    for key in metrics.keys():
        tf.summary.scalar(key, metrics[key][1])
