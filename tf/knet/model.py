#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import struct
import tensorflow as tf
import numpy as np

import input_data
import custom_ops

PADDING_ID = 0

NCE_WEIGHTS_NAME = 'nce_weights'
NCE_BIASES_NAME = 'nce_biases'

# filename must end with .npy
SAVE_NCE_WEIGHTS_NAME = 'nce_weights.npy'
SAVE_NCE_BIASES_NAME = 'nce_biases.npy'

NCE_PARAM_NAMES = [
    SAVE_NCE_WEIGHTS_NAME,
    SAVE_NCE_BIASES_NAME,
]

OPTIMIZE_LEVEL_ZERO = 0
OPTIMIZE_LEVEL_SAVED_NCE_PARAMS = 1
OPTIMIZE_LEVEL_OPENBLAS_TOP_K = 2

ALL_OPTIMIZE_LEVELS = [
    OPTIMIZE_LEVEL_ZERO,
    OPTIMIZE_LEVEL_SAVED_NCE_PARAMS,
    OPTIMIZE_LEVEL_OPENBLAS_TOP_K,
]


def knet_model(features, labels, mode, params):
    """ build model graph """

    n_classes = params['n_classes']
    embedding_dim = params['embedding_dim']
    lr = params['learning_rate']
    num_sampled = params['num_sampled']
    feature_columns = params['feature_columns']
    recall_k = params['recall_k']
    dict_dir = params['dict_dir']
    optimize_level = params['optimize_level']
    nce_params_dir = params['nce_params_dir']

    embeddings = get_embeddings(n_classes, embedding_dim)
    nce_weights, nce_biases = get_nce_weights_and_biases(n_classes,
                                                         embedding_dim)

    # construct network
    input_layer = tf.feature_column.input_layer(features, feature_columns)
    nonzeros = tf.count_nonzero(input_layer, 1, keepdims=True)
    nonzeros = tf.maximum(nonzeros, 1)  # avoid divide zero
    embeds = mask_padding_embedding_lookup(embeddings, embedding_dim,
                                           input_layer, PADDING_ID)
    embeds_sum = tf.reduce_sum(embeds, 1)
    embeds_mean = embeds_sum / tf.cast(nonzeros, tf.float32)

    hidden = embeds_mean
    for units in params['hidden_units']:
        hidden = tf.layers.dense(hidden, units=units, activation=tf.nn.relu,
                                 name="fc_{}".format(units))
    user_vector = tf.layers.dense(hidden, embedding_dim, activation=None,
                                  name="user_vector")

    # Compute logits (just for train Mode, for metric summary, no optimize).
    train_logits = tf.matmul(user_vector, nce_weights, transpose_b=True,
                             name="matmul_logits")
    train_logits = tf.nn.bias_add(train_logits, nce_biases,
                                  name="bias_add_logits")
    train_scores, train_ids = tf.nn.top_k(train_logits, recall_k,
                                          name="top_k_{}".format(recall_k))

    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        with tf.name_scope("PredictMode"):
            if optimize_level == OPTIMIZE_LEVEL_SAVED_NCE_PARAMS:
                tf.logging.info("Use OPTIMIZE_LEVEL_SAVED_NCE_PARAMS")
                scores, ids, _ = optimize_level_saved_nce_params(
                    nce_params_dir, user_vector, recall_k)
            elif optimize_level == OPTIMIZE_LEVEL_OPENBLAS_TOP_K:
                tf.logging.info("Use OPTIMIZE_LEVEL_OPENBLAS_TOP_K")
                scores, ids = optimize_level_openblas_top_k(
                    nce_params_dir, user_vector, recall_k)

        table = create_index_to_string_table(dict_dir)
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

    train_metrics = get_metrics(labels, train_ids, recall_k)
    add_metrics_summary(train_metrics)

    if mode == tf.estimator.ModeKeys.EVAL:
        with tf.name_scope("EvalMode"):
            if optimize_level != OPTIMIZE_LEVEL_ZERO:
                # Eval mode only surpport optimize_level_saved_nce_params
                tf.logging.info("Use OPTIMIZE_LEVEL_SAVED_NCE_PARAMS to eval")
                scores, ids, logits = optimize_level_saved_nce_params(
                    nce_params_dir, user_vector, recall_k)
            else:
                # No optimize
                scores = train_scores
                ids = train_ids

            eval_metrics = get_metrics(labels, ids, recall_k)
            eval_loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        return tf.estimator.EstimatorSpec(
            mode,
            loss=tf.constant(0),  # don't evaluate loss
            eval_metric_ops=eval_metrics)

    # Create training op.
    nce_loss = tf.nn.nce_loss(weights=nce_weights,
                              biases=nce_biases,
                              labels=labels,
                              inputs=user_vector,
                              num_sampled=num_sampled,
                              num_classes=n_classes,
                              partition_strategy="div",
                              name="nce_loss")
    nce_loss = tf.reduce_mean(nce_loss, name="mean_nce_loss")

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(nce_loss,
                                  global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=nce_loss, train_op=train_op)


def get_nce_weights_and_biases(n_classes, embedding_dim):
    """Get nce weights and biases variables."""

    with tf.variable_scope("nce_layer", reuse=tf.AUTO_REUSE):
        # Construct the variables for the NCE loss
        nce_weights = tf.get_variable(
            NCE_WEIGHTS_NAME,
            initializer=tf.truncated_normal(
                [n_classes, embedding_dim],
                stddev=1.0 / math.sqrt(embedding_dim)))
        nce_biases = tf.get_variable(NCE_BIASES_NAME,
                                     initializer=tf.zeros([n_classes]),
                                     trainable=False)
    return nce_weights, nce_biases


def get_embeddings(n_classes, embedding_dim):
    """Get embeddings variables."""

    with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
        embeddings = tf.get_variable(
            "embeddings",
            initializer=tf.random_uniform([n_classes, embedding_dim],
                                          -1.0, 1.0))
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
        output = tf.nn.embedding_lookup(
            embeddings, tf.cast(input, tf.int32, name="lookup_idx_cast"),
            name="embedding_lookup")
    return output


def get_metrics(labels, ids, recall_k):
    """Get metrics dict."""

    predicted = ids[:, :1]
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted)
    recall_at_top_k = tf.metrics.recall_at_top_k(
        labels=labels, predictions_idx=ids, k=recall_k)
    precision_at_top_k = tf.metrics.precision_at_top_k(
        labels=labels, predictions_idx=ids, k=recall_k)
    metrics = {'accuracy': accuracy,
               'recall_at_top_{}'.format(recall_k): recall_at_top_k,
               'precision_at_top_{}'.format(recall_k): precision_at_top_k}
    return metrics


def add_metrics_summary(metrics):
    """Add metrics to tensorboard."""

    for key in metrics.keys():
        tf.summary.scalar(key, metrics[key][1])


def create_index_to_string_table(dict_dir):
    """Create index to string[rowkey] table."""

    dict_words_path = os.path.join(dict_dir, input_data.DICT_WORDS)
    words = [line.strip() for line in open(dict_words_path)
             if line.strip() != '']
    words.insert(0, '')
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        mapping=words,
        default_value='',
        name="index_to_string")
    return table


def optimize_level_saved_nce_params(nce_params_dir, user_vector, recall_k):
    """Optimize top-K with pre-saved nce weights and biases. Decrease
    delay from 200ms to 30ms per request.
    """

    (saved_nce_weights,
     saved_nce_biases) = load_model_nce_params(nce_params_dir)
    transpose_saved_nce_weights = tf.convert_to_tensor(
        saved_nce_weights.transpose(), dtype=tf.float32,
        name='transpose_saved_nce_weights')
    saved_nce_biases = tf.convert_to_tensor(
        saved_nce_biases, dtype=tf.float32,
        name='saved_nce_biases')
    logits = tf.matmul(
        user_vector, transpose_saved_nce_weights,
        name="matmul_logits")
    logits = tf.nn.bias_add(
        logits, saved_nce_biases, name="bias_add_logits")
    scores, ids = tf.nn.top_k(
        logits, recall_k, name="top_k_{}".format(recall_k))
    return (scores, ids, logits)


def optimize_level_openblas_top_k(nce_params_dir, user_vector, recall_k):
    """Optimize using openblas to calculate top-K."""

    (saved_nce_weights,
     saved_nce_biases) = load_model_nce_params(nce_params_dir)
    saved_nce_weights = tf.make_tensor_proto(saved_nce_weights)
    saved_nce_biases = tf.make_tensor_proto(saved_nce_biases)

    scores, ids = custom_ops.openblas_top_k(
        input=user_vector, k=recall_k,
        weights=saved_nce_weights,
        biases=saved_nce_biases)
    return scores, ids


def get_model_nce_weights_and_biases(model):
    """Get nce weights and biases variables from estimator model"""

    nce_weights = model.get_variable_value('nce_layer/' + NCE_WEIGHTS_NAME)
    nce_biases = model.get_variable_value('nce_layer/' + NCE_BIASES_NAME)
    return (nce_weights, nce_biases)


def save_model_nce_params(model, nce_params_dir):
    """Save model nce weights and biases variables."""

    nce_weights, nce_biases = get_model_nce_weights_and_biases(model)
    tf.logging.info('save nce_weights = \n{}'.format(nce_weights))
    tf.logging.info('save nce_biases = \n{}'.format(nce_biases))
    save_weights_path = os.path.join(nce_params_dir, SAVE_NCE_WEIGHTS_NAME)
    save_biases_path = os.path.join(nce_params_dir, SAVE_NCE_BIASES_NAME)
    np.save(save_weights_path, nce_weights)
    np.save(save_biases_path, nce_biases)


def load_model_nce_params(nce_params_dir):
    """Load pre-saved model nce weights and biases."""

    save_weights_path = os.path.join(nce_params_dir, SAVE_NCE_WEIGHTS_NAME)
    save_biases_path = os.path.join(nce_params_dir, SAVE_NCE_BIASES_NAME)
    nce_weights = np.load(save_weights_path)
    nce_biases = np.load(save_biases_path)
    tf.logging.info('load nce_weights = \n{}'.format(nce_weights))
    tf.logging.info('load nce_biases = \n{}'.format(nce_biases))

    return (nce_weights, nce_biases)


def save_numpy_float_array(array, filename):
    """Save numpy float array to file."""

    with open(filename, 'wb') as f:
        for d in array.shape:
            f.write(struct.pack('<q', d))

        fl = array.flat
        for v in fl:
            f.write(struct.pack('<f', v))
