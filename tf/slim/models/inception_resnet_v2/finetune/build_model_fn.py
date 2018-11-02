#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

import model_keys
from nets import inception


_global_learning_rate = 0.001


class _RestoreHook(tf.train.SessionRunHook):
    """_Restore from pretrained checkpoint."""

    def __init__(self, init_fn):
        self.init_fn = init_fn

    def after_create_session(self, session, coord=None):
        if session.run(tf.train.get_or_create_global_step()) == 0:
            self.init_fn(session)


def set_global_learning_rate(rate):
    global _global_learning_rate
    _global_learning_rate = rate


def get_global_learning_rate():
    global _global_learning_rate
    return _global_learning_rate


# build inception_resnet_v2
def inception_resnet_v2_model_fn(features, labels, mode, params):
    """Build model graph."""

    tf.summary.image('images', features[model_keys.DATA_COL])

    opts = params['opts']
    inputs = features[model_keys.DATA_COL]

    training = (mode == tf.estimator.ModeKeys.TRAIN)

    with slim.arg_scope(inception.inception_resnet_v2_arg_scope(
            weight_decay=opts.l2_regularizer,
            batch_norm_decay=opts.batch_norm_decay,
            batch_norm_epsilon=opts.batch_norm_epsilon,
            activation_fn=tf.nn.relu)):
        logits, end_points = inception.inception_resnet_v2(
            inputs,
            num_classes=opts.num_classes,
            is_training=training,
            dropout_keep_prob=opts.dropout_keep_prob,
            reuse=None,
            create_aux_logits=opts.create_aux_logits)

    training_hooks = []
    training_hooks.append(create_restore_hook(opts))

    score = tf.nn.softmax(logits)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return create_predict_estimator_spec(mode, score, labels, params)

    if mode == tf.estimator.ModeKeys.EVAL:
        return create_eval_estimator_spec(mode, logits, labels, params)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return create_train_estimator_spec(
            mode, logits, labels, params, training_hooks)


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


def create_train_estimator_spec(mode, logits, labels, params, training_hooks):
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

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for bn
    train_op = tf.contrib.training.create_train_op(
        total_loss=loss,
        optimizer=optimizer,
        global_step=global_step,
        update_ops=update_ops,
        variables_to_train=get_finetune_trainable_variables(opts),
        transform_grads_fn=None,
        summarize_gradients=True,
        aggregation_method=None,
        colocate_gradients_with_ops=False,
        check_numerics=True)

    return tf.estimator.EstimatorSpec(
        mode, loss=loss, train_op=train_op, training_hooks=training_hooks)


def add_metrics_summary(metrics):
    """Add metrics to tensorboard."""

    for key in metrics.keys():
        tf.summary.scalar(key, metrics[key][1])


def create_restore_hook(opts):
    variables_to_restore = slim.get_variables_to_restore(
        exclude=opts.exclude_restore_layers)
    tf.logging.info('Restore variables: ')
    for var in variables_to_restore:
        tf.logging.info(var)
    init_fn = slim.assign_from_checkpoint_fn(
        opts.pretrained_weights_path,
        variables_to_restore,
        ignore_missing_vars=False)

    tf.logging.info('Global trainable variables: ')
    for var in slim.get_trainable_variables():
        tf.logging.info(var)

    return _RestoreHook(init_fn)


def get_finetune_trainable_variables(opts):
    trainable_variables = []
    for scope in opts.train_layers:
        trainable_variables.extend(slim.get_variables(scope))

    tf.logging.info("Finetune variables:")
    for var in trainable_variables:
        tf.logging.info(var)

    return trainable_variables
