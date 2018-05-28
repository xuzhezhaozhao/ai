#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import input_data
import math
import os

PADDING_ID = 0


def get_nce_weights_and_biases(n_classes, embedding_dim):
    """ Construct the variables for the NCE loss """

    with tf.variable_scope("nce_layer", reuse=tf.AUTO_REUSE):
        # Construct the variables for the NCE loss
        nce_weights = tf.get_variable(
            "nce_weights",
            initializer=tf.truncated_normal(
                [n_classes, embedding_dim],
                stddev=1.0 / math.sqrt(embedding_dim)))
        nce_biases = tf.get_variable("nce_biases",
                                     initializer=tf.zeros([n_classes]))
    return nce_weights, nce_biases


def get_embeddings(n_classes, embedding_dim):
    """ Construct the variable for the embeddings """

    with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
        embeddings = tf.get_variable(
            "embeddings",
            initializer=tf.random_uniform([n_classes, embedding_dim],
                                          -1.0, 1.0))
    return embeddings


def mask_padding_embedding_lookup(embeddings, embedding_dim,
                                  input, padding_id):
    """ ref(@ay27): https://github.com/tensorflow/tensorflow/issues/2373 """

    mask_padding_zero_op = tf.scatter_update(
        embeddings, padding_id, tf.zeros([embedding_dim], dtype=tf.float32),
        name="mask_padding_zero_op")
    with tf.control_dependencies([mask_padding_zero_op]):
        output = tf.nn.embedding_lookup(
            embeddings, tf.cast(input, tf.int32, name="lookup_idx_cast"),
            name="embedding_lookup")
    return output


def knet_model(features, labels, mode, params):
    """ build model graph """

    n_classes = params['n_classes']
    embedding_dim = params['embedding_dim']
    lr = params['learning_rate']
    num_sampled = params['num_sampled']
    feature_columns = params['feature_columns']
    recall_k = params['recall_k']
    dict_dir = params['dict_dir']

    embeddings = get_embeddings(n_classes, embedding_dim)
    nce_weights, nce_biases = get_nce_weights_and_biases(n_classes,
                                                         embedding_dim)

    # construct network
    net = tf.feature_column.input_layer(features, feature_columns)
    net = mask_padding_embedding_lookup(embeddings, embedding_dim,
                                        net, PADDING_ID)
    net = tf.reduce_mean(net, 1, name="mean")

    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu,
                              name="hidden_{}".format(units))
    net = tf.layers.dense(net, embedding_dim, activation=None,
                          name="knet_output")

    # TODO(zhezhaoxu) It's too slow when used in online serving, write custom
    # op to  optimize. We can pre-calculate transpose(nce_weights) and use
    # openblas to calculate matmul when serving
    # Compute logits (1 per class).
    logits = tf.matmul(net, nce_weights, transpose_b=True,
                       name="matmul_logits")
    logits = tf.nn.bias_add(logits, nce_biases, name="bias_add_logits")

    # Optimaize, don't need calculate softmax
    # probabilities = tf.nn.softmax(logits)
    scores, ids = tf.nn.top_k(logits, recall_k,
                              name="top_k_{}".format(recall_k))

    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        dict_words_path = os.path.join(dict_dir, input_data.DICT_WORDS)
        words = [line.strip() for line in open(dict_words_path)
                 if line.strip() != '']
        words.insert(0, '')
        table = tf.contrib.lookup.index_to_string_table_from_tensor(
            mapping=words,
            default_value='',
            name="index_to_string")

        predictions = {
            'class_ids': ids,
            'scores': scores,
            'words': table.lookup(tf.cast(ids, tf.int64))
        }
        export_outputs = {
            'predicts': tf.estimator.export.PredictOutput(
                outputs={
                    'class_ids': ids,
                    'scores': scores,
                    'words': table.lookup(tf.cast(ids, tf.int64))
                }
            )
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions,
                                          export_outputs=export_outputs)

    # Compute nce_loss.
    nce_loss = tf.nn.nce_loss(weights=nce_weights,
                              biases=nce_biases,
                              labels=labels,
                              inputs=net,
                              num_sampled=num_sampled,
                              num_classes=n_classes,
                              partition_strategy="div",
                              name="nce_loss")
    nce_loss = tf.reduce_mean(nce_loss, name="mean_nce_loss")

    # Compute evaluation metrics.
    predicted = tf.argmax(logits, 1)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted)
    recall_at_top_k = tf.metrics.recall_at_top_k(
        labels=labels, predictions_idx=ids, k=recall_k)
    precision_at_top_k = tf.metrics.precision_at_top_k(
        labels=labels, predictions_idx=ids, k=recall_k)
    metrics = {'accuracy': accuracy,
               'recall_at_top_k': recall_at_top_k,
               'precision_at_top_k': precision_at_top_k}

    # Don't summary to speedup?
    tf.summary.scalar('accuracy', accuracy[1])
    tf.summary.scalar('recall_at_top_{}'.format(recall_k), recall_at_top_k[1])
    tf.summary.scalar('precision_at_top_{}'.format(recall_k),
                      precision_at_top_k[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(nce_loss,
                                  global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=nce_loss, train_op=train_op)
