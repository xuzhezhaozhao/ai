#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


class _RestoreHook(tf.train.SessionRunHook):
    """_Restore from pretrained checkpoint."""

    def __init__(self, init_fn):
        self.init_fn = init_fn

    def after_create_session(self, session, coord=None):
        if session.run(tf.train.get_or_create_global_step()) == 0:
            self.init_fn(session)


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


def create_predict_estimator_spec(mode, logits, labels, params):
    """Create predict EstimatorSpec."""

    score = tf.nn.softmax(logits)
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

    opts = params['opts']

    training_hooks = []
    training_hooks.append(create_restore_hook(opts))

    tf.summary.scalar('train_batch_accuracy', tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1)),
                tf.float32)))

    opts = params['opts']
    global_step = tf.train.get_global_step()
    loss = get_loss(logits, labels, 'train')
    learning_rate = opts.learning_rate
    tf.logging.info("global learning rate = {}".format(learning_rate))
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = configure_optimizer(learning_rate, opts)

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


def create_estimator_spec(mode, logits, labels, params):
    if mode == tf.estimator.ModeKeys.PREDICT:
        return create_predict_estimator_spec(mode, logits, labels, params)

    if mode == tf.estimator.ModeKeys.EVAL:
        return create_eval_estimator_spec(mode, logits, labels, params)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return create_train_estimator_spec(mode, logits, labels, params)


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
        trainable_variables.extend(slim.get_trainable_variables(scope))

    tf.logging.info("Finetune variables:")
    for var in trainable_variables:
        tf.logging.info(var)

    return trainable_variables


def configure_optimizer(learning_rate, opts):
    """Configures the optimizer used for training."""

    if opts.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=opts.adadelta_rho,
            epsilon=opts.opt_epsilon)
    elif opts.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=opts.adagrad_initial_accumulator_value)
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
        raise ValueError('Optimizer [%s] was not recognized' % opts.optimizer)
    return optimizer
