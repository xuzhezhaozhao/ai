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

import model_keys
import custom_ops


def knet_model_fn(features, labels, mode, params):
    """Build model graph."""

    feature_columns = params['feature_columns']
    predict_feature_columns = params['predict_feature_columns']
    n_classes = params['n_classes']
    ntokens = params['ntokens']
    opts = params['opts']

    embedding_dim = opts.dim
    lr = opts.lr
    hidden_units = opts.hidden_units
    num_sampled = opts.num_sampled
    recall_k = opts.recall_k
    dict_dir = opts.dict_dir
    optimize_level = opts.optimize_level
    use_subset = opts.use_subset
    dropout = opts.dropout
    ntargets = opts.ntargets
    train_nce_biases = opts.train_nce_biases
    batch_size = opts.batch_size
    optimizer_type = opts.optimizer_type
    num_in_graph_replication = opts.num_in_graph_replication

    embeddings = get_embeddings(n_classes, embedding_dim)
    (nce_weights,
     nce_biases) = get_nce_weights_and_biases(
         n_classes, embedding_dim, train_nce_biases)

    if mode == tf.estimator.ModeKeys.PREDICT:
        input_layer = tf.feature_column.input_layer(
            features, predict_feature_columns)
    else:
        input_layer = tf.feature_column.input_layer(features, feature_columns)

    nonzeros = tf.count_nonzero(input_layer, 1, keepdims=True)  # [batch, 1]
    nonzeros = tf.maximum(nonzeros, 1)  # avoid divide zero
    embeds = mask_padding_embedding_lookup(embeddings, embedding_dim,
                                           input_layer, model_keys.PADDING_ID)
    embeds_sum = tf.reduce_sum(embeds, 1)
    embeds_mean = embeds_sum / tf.cast(nonzeros, tf.float32)

    hidden = embeds_mean
    for index, units in enumerate(hidden_units):
        hidden = tf.layers.dense(
            hidden, units=units, activation=tf.nn.relu,
            name="fc{}_{}".format(index, units), reuse=tf.AUTO_REUSE)
        if dropout > 0:
            training = (mode == tf.estimator.ModeKeys.TRAIN)
            hidden = tf.layers.dropout(
                hidden, dropout, training=training,
                name="dropout{}_{}".format(index, units))
    user_vector = tf.layers.dense(hidden, embedding_dim, activation=None,
                                  name="user_vector")

    # Compute logits(just for train Mode, for metric summary, no optimize).
    train_logits = tf.matmul(user_vector, nce_weights, transpose_b=True,
                             name="matmul_logits")
    train_logits = tf.nn.bias_add(train_logits, nce_biases,
                                  name="bias_add_logits")
    train_scores, train_ids = tf.nn.top_k(train_logits, recall_k,
                                          name="top_k_{}".format(recall_k))

    if mode == tf.estimator.ModeKeys.PREDICT:
        with tf.name_scope("PredictMode"):
            if optimize_level == model_keys.OPTIMIZE_LEVEL_SAVED_NCE_PARAMS:
                tf.logging.info("Use OPTIMIZE_LEVEL_SAVED_NCE_PARAMS")
                scores, ids, _ = optimize_level_saved_nce_params(
                    dict_dir, user_vector, recall_k, use_subset)
            elif optimize_level == model_keys.OPTIMIZE_LEVEL_OPENBLAS_TOP_K:
                tf.logging.info("Use OPTIMIZE_LEVEL_OPENBLAS_TOP_K")
                scores, ids = optimize_level_openblas_top_k(
                    dict_dir, user_vector, recall_k, use_subset)

        table = create_index_to_string_table(dict_dir, use_subset)
        predictions = {
            'class_ids': ids,
            'scores': scores,
            'words': table.lookup(tf.cast(ids, tf.int64)),
            'num_in_dict': features[model_keys.NUM_IN_DICT_COL]
        }
        export_outputs = {
            'predicts': tf.estimator.export.PredictOutput(
                outputs={
                    'scores': scores,
                    'words': table.lookup(tf.cast(ids, tf.int64)),
                    'num_in_dict': features[model_keys.NUM_IN_DICT_COL]
                }
            )
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions,
                                          export_outputs=export_outputs)

    train_metrics = get_metrics(labels, train_ids, recall_k, ntargets)
    add_metrics_summary(train_metrics)

    if mode == tf.estimator.ModeKeys.EVAL:
        with tf.name_scope("EvalMode"):
            # TODO Don't optimize because of distribute model
            if False:
                # Eval mode only surpport optimize_level_saved_nce_params
                tf.logging.info("Use OPTIMIZE_LEVEL_SAVED_NCE_PARAMS to eval")
                scores, ids, logits = optimize_level_saved_nce_params(
                    dict_dir, user_vector, recall_k, use_subset)
            else:
                # No optimize
                scores = train_scores
                ids = train_ids

            eval_metrics = get_metrics(labels, ids, recall_k, ntargets)
        return tf.estimator.EstimatorSpec(
            mode,
            loss=tf.constant(0),  # don't evaluate loss
            eval_metric_ops=eval_metrics)

    replica_losses = create_losses_in_graph_replications(
        num_in_graph_replication=num_in_graph_replication,
        batch_size=batch_size,
        weights=nce_weights,
        biases=nce_biases,
        labels=labels,
        inputs=user_vector,
        num_sampled=num_sampled,
        num_classes=n_classes,
        num_true=ntargets,
        partition_strategy="div",
        name="nce_loss")
    loss = sum(replica_losses)

    assert mode == tf.estimator.ModeKeys.TRAIN

    if optimizer_type == model_keys.OptimizerType.ADA:
        optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    elif optimizer_type == model_keys.OptimizerType.ADADELTA:
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr)
    elif optimizer_type == model_keys.OptimizerType.ADAM:
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    elif optimizer_type == model_keys.OptimizerType.SGD:
        processed_tokens = features[model_keys.TOKENS_COL][0][0]
        tf.summary.scalar("processed_tokens", processed_tokens)
        new_lr = lr * (1.0 - (tf.cast(processed_tokens, tf.float32)
                              / tf.cast(ntokens, tf.float32)))
        new_lr = tf.maximum(new_lr, 1e-6)
        tf.summary.scalar("lr", new_lr)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=new_lr)
    else:
        raise ValueError('OptimizerType "{}" not surpported.'
                         .format(optimizer_type))

    ops = []
    for loss in replica_losses:
        op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        ops.append(op)
    train_op = tf.group(*ops)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def get_nce_weights_and_biases(n_classes, embedding_dim, train_nce_biases):
    """Get nce weights and biases variables."""

    with tf.variable_scope("nce_layer", reuse=tf.AUTO_REUSE):
        nce_weights = tf.get_variable(
            model_keys.NCE_WEIGHTS_NAME,
            initializer=tf.truncated_normal(
                [n_classes, embedding_dim],
                stddev=1.0 / math.sqrt(embedding_dim)))
        nce_biases = tf.get_variable(
            model_keys.NCE_BIASES_NAME, initializer=tf.zeros([n_classes]),
            trainable=train_nce_biases)
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


def get_metrics(labels, ids, recall_k, ntargets):
    """Get metrics dict."""

    predicted = ids[:, :ntargets]
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


def create_index_to_string_table(dict_dir, use_subset):
    """Create index to string[rowkey] table."""

    dict_words_name = (model_keys.DICT_WORDS_SUBSET if use_subset
                       else model_keys.DICT_WORDS)
    dict_words_path = os.path.join(dict_dir, dict_words_name)
    words = [line.strip() for line in open(dict_words_path)
             if line.strip() != '']
    if not use_subset:
        words.insert(0, '')
    tf.logging.info("create_index_to_string_table size={}".format(len(words)))
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        mapping=words,
        default_value='',
        name="index_to_string_table")
    return table


def optimize_level_saved_nce_params(
        dict_dir, user_vector, recall_k, use_subset):
    """Optimize top-K with pre-saved nce weights and biases. Decrease
    delay from 200ms to 30ms per request.
    """

    (saved_nce_weights,
     saved_nce_biases) = load_model_nce_params(dict_dir, use_subset)
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


def optimize_level_openblas_top_k(dict_dir, user_vector, recall_k, use_subset):
    """Optimize using openblas to calculate top-K."""

    (saved_nce_weights,
     saved_nce_biases) = load_model_nce_params(dict_dir, use_subset)
    saved_nce_weights = tf.make_tensor_proto(saved_nce_weights)
    saved_nce_biases = tf.make_tensor_proto(saved_nce_biases)

    scores, ids = custom_ops.openblas_top_k(
        input=user_vector, k=recall_k,
        weights=saved_nce_weights,
        biases=saved_nce_biases)
    return scores, ids


def get_model_nce_weights_and_biases(model):
    """Get nce weights and biases variables from estimator model"""

    nce_weights = model.get_variable_value(
        'nce_layer/' + model_keys.NCE_WEIGHTS_NAME)
    nce_biases = model.get_variable_value(
        'nce_layer/' + model_keys.NCE_BIASES_NAME)
    return (nce_weights, nce_biases)


def save_model_nce_params(model, dict_dir):
    """Save model nce weights and biases variables."""

    nce_weights, nce_biases = get_model_nce_weights_and_biases(model)
    tf.logging.info('save nce_weights = \n{}'.format(nce_weights))
    tf.logging.info('save nce_biases = \n{}'.format(nce_biases))
    save_weights_path = os.path.join(
        dict_dir, model_keys.SAVE_NCE_WEIGHTS_NAME)
    save_biases_path = os.path.join(dict_dir, model_keys.SAVE_NCE_BIASES_NAME)
    np.save(save_weights_path, nce_weights)
    np.save(save_biases_path, nce_biases)


def load_model_nce_params(dict_dir, use_subset):
    """Load pre-saved model nce weights and biases."""

    nce_weights_name = (model_keys.SAVE_NCE_WEIGHTS_SUBSET_NAME if use_subset
                        else model_keys.SAVE_NCE_WEIGHTS_NAME)
    nce_biases_name = (model_keys.SAVE_NCE_BIASES_SUBSET_NAME if use_subset
                       else model_keys.SAVE_NCE_BIASES_NAME)

    save_weights_path = os.path.join(dict_dir, nce_weights_name)
    save_biases_path = os.path.join(dict_dir, nce_biases_name)
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


def is_in_subset(word):
    """Check wheather rowkey is video, yes return True, or False."""

    if len(word) > 5:
        if word[-2] == 'a' or word[-2] == 'b':
            return True
    return False


def filter_and_save_subset(dict_dir):
    save_weights_path = os.path.join(
        dict_dir, model_keys.SAVE_NCE_WEIGHTS_NAME)
    save_biases_path = os.path.join(dict_dir, model_keys.SAVE_NCE_BIASES_NAME)
    nce_weights = np.load(save_weights_path)
    nce_biases = np.load(save_biases_path)
    dict_words_path = os.path.join(dict_dir, model_keys.DICT_WORDS)
    words = [line.strip() for line in open(dict_words_path)
             if line.strip() != '']

    subset_words = [w for w in words if is_in_subset(w)]
    num_in_subset = len(subset_words)
    dim = nce_weights.shape[-1]

    subset_weights = np.zeros([num_in_subset, dim], dtype=np.float32)
    subset_biases = np.zeros([num_in_subset], dtype=np.float32)
    tf.logging.info("subset length = {}".format(num_in_subset))

    subset_index = 0
    for index, word in enumerate(words):
        if is_in_subset(word):
            # index plus one because of padding
            subset_weights[subset_index] = nce_weights[index + 1]
            subset_biases[subset_index] = nce_biases[index + 1]
            subset_index += 1

    to_save_subset_words = reduce(lambda w1, w2: w1 + '\n' + w2, subset_words)

    with open(os.path.join(dict_dir, model_keys.DICT_WORDS_SUBSET), 'w') as f:
        f.write(to_save_subset_words)

    np.save(os.path.join(dict_dir, model_keys.SAVE_NCE_WEIGHTS_SUBSET_NAME),
            subset_weights)
    np.save(os.path.join(dict_dir, model_keys.SAVE_NCE_BIASES_SUBSET_NAME),
            subset_biases)


def create_losses_in_graph_replications(num_in_graph_replication, batch_size,
                                          weights, biases, labels, inputs,
                                          num_sampled, num_classes, num_true,
                                          partition_strategy, name):
    losses = []
    for i in range(num_in_graph_replication):
        repli_labels = labels[i * batch_size:(i + 1) * batch_size, :]
        repli_inputs = inputs[i * batch_size:(i + 1) * batch_size, :]
        nce_loss = tf.nn.nce_loss(weights=weights,
                                  biases=biases,
                                  labels=repli_labels,
                                  inputs=repli_inputs,
                                  num_sampled=num_sampled,
                                  num_classes=num_classes,
                                  num_true=num_true,
                                  partition_strategy=partition_strategy,
                                  name=name)
        nce_loss = tf.reduce_mean(nce_loss, name="mean_nce_loss")
        losses.append(nce_loss)

    return losses
