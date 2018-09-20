#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model_keys


def krank_model_fn(features, labels, mode, params):
    """Build model graph."""

    opts = params['opts']
    rowkey_embedding_dim = opts.rowkey_embedding_dim

    rowkey_embeddings = get_rowkey_embeddings(params)
    positive_records = features[model_keys.POSITIVE_RECORDS_COL]
    negative_records = features[model_keys.NEGATIVE_RECORDS_COL]
    targets = features[model_keys.TARGETS_COL]

    positive_records.set_shape([None, opts.train_ws])
    negative_records.set_shape([None, opts.train_ws])
    targets.set_shape([None])

    positive_nonzeros = tf.count_nonzero(positive_records, 1, keepdims=True)
    positive_nonzeros = tf.maximum(positive_nonzeros, 1)
    positive_embeds = mask_padding_embedding_lookup(
        rowkey_embeddings, rowkey_embedding_dim, positive_records, 0)
    positive_embeds_sum = tf.reduce_sum(positive_embeds, 1)
    positive_embeds_mean = positive_embeds_sum / tf.cast(positive_nonzeros,
                                                         tf.float32)

    negative_nonzeros = tf.count_nonzero(negative_records, 1, keepdims=True)
    negative_nonzeros = tf.maximum(negative_nonzeros, 1)
    negative_embeds = mask_padding_embedding_lookup(
        rowkey_embeddings, rowkey_embedding_dim, negative_records, 0)
    negative_embeds_sum = tf.reduce_sum(negative_embeds, 1)
    negative_embeds_mean = negative_embeds_sum / tf.cast(negative_nonzeros,
                                                         tf.float32)

    targets_embeds = mask_padding_embedding_lookup(
        rowkey_embeddings, rowkey_embedding_dim, targets, 0)

    concat_features = [positive_embeds_mean,
                       negative_embeds_mean,
                       targets_embeds]

    input_layer = tf.concat(concat_features, 1)

    training = (mode == tf.estimator.ModeKeys.TRAIN)
    lr_dim = input_layer.shape[-1].value

    with tf.name_scope('hidden_layers'):
        hidden = input_layer
        for index, units in enumerate(opts.hidden_units):
            hidden = tf.layers.dense(hidden, units=units, activation=None,
                                     name='fc{}_{}'.format(index, units),
                                     reuse=tf.AUTO_REUSE)
            hidden = batch_normalization(hidden, training,
                                         name='bn{}_{}'.format(index, units))
            hidden = tf.nn.relu(hidden)
            hidden = tf.layers.dropout(
                hidden, opts.dropout, training=training,
                name='dropout{}_{}'.format(index, units))
            lr_dim = units

    lr_weights, lr_biases = get_lr_weights_and_biases(params, lr_dim)

    logits = tf.reduce_sum(hidden * lr_weights, 1)
    logits = logits + lr_biases
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(labels, tf.float32), logits=logits)
    loss = tf.reduce_mean(loss)

    global_step = tf.train.get_global_step()
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=opts.lr,
                                           beta1=0.9,
                                           beta2=0.999,
                                           epsilon=1e-08)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for bn
        with tf.control_dependencies(update_ops):
            gradients, variables = zip(*optimizer.compute_gradients(
                loss, gate_gradients=tf.train.Optimizer.GATE_GRAPH))
            train_op = optimizer.apply_gradients(
                zip(gradients, variables), global_step=global_step)

            for var, grad in zip(variables, gradients):
                tf.summary.histogram(var.name.replace(':', '_') + '/gradient',
                                     grad)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        score = tf.nn.sigmoid(logits)
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=tf.to_int32(score > 0.5))
        auc = tf.metrics.auc(labels=labels,
                             predictions=score,
                             num_thresholds=1000)

        metrics = {
            'accuracy': accuracy,
            'auc': auc,
        }

        return tf.estimator.EstimatorSpec(mode, loss=loss,
                                          eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'logits': logits,
            'score': tf.nn.sigmoid(logits),
        }

        export_outputs = {
            'predicts': tf.estimator.export.PredictOutput(
                outputs={
                    'logits': logits,
                    'score': tf.nn.sigmoid(logits),
                }
            )
        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions,
                                          export_outputs=export_outputs)


def get_rowkey_embeddings(params):
    """Get rowkey embeddings variables."""

    opts = params['opts']
    num_rowkey = opts.num_rowkey
    dim = opts.rowkey_embedding_dim

    with tf.variable_scope("rowkey_embeddings", reuse=tf.AUTO_REUSE):
        init_width = 1.0 / dim
        embeddings = tf.get_variable(
            "embeddings",
            initializer=tf.random_uniform([num_rowkey, dim],
                                          -init_width, init_width))
    return embeddings


def mask_padding_embedding_lookup(embeddings, embedding_dim,
                                  input, padding_id):
    """ mask padding tf.nn.embedding_lookup.
    padding_id must be zero.

    ref(@ay27): https://github.com/tensorflow/tensorflow/issues/2373
    """

    assert padding_id == 0

    mask_padding_zero_op = tf.scatter_update(
        embeddings, padding_id, tf.zeros([embedding_dim], dtype=tf.float32),
        name="mask_padding_zero_op")
    with tf.control_dependencies([mask_padding_zero_op]):
        output = tf.nn.embedding_lookup(embeddings, input,
                                        name="embedding_lookup")
    return output


def batch_normalization(input, training, name):
    """batch normalization layer."""

    bn = tf.layers.batch_normalization(
        input, axis=1, training=training,
        scale=False, trainable=True,
        name=name, reuse=tf.AUTO_REUSE)
    return bn


def get_lr_weights_and_biases(params, dim):
    """Get logistic regression weights and biases."""

    with tf.variable_scope('logiistic_regression', reuse=tf.AUTO_REUSE):
        weights = tf.get_variable('weights', initializer=tf.zeros([dim]))
        biases = tf.get_variable('biases', initializer=[1.0], dtype=tf.float32)

        return weights, biases
