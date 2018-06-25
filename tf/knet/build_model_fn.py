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


_optimizer_id = 0


def knet_model_fn(features, labels, mode, params):
    """Build model graph."""

    with tf.name_scope("model_fn_{}".format(_optimizer_id)):
        (embeddings, nce_weights, nce_biases) = get_nce_variables(params)

        input_layer = create_input_layer(mode, features, params, embeddings)
        user_vector = create_hidden_layer(mode, input_layer, params)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return create_predict_estimator_spec(
                mode, user_vector, features, params)

        _, top_k_ids = compute_top_k(
            nce_weights, nce_biases, user_vector, params)
        metrics = get_metrics(labels, top_k_ids, params)
        add_metrics_summary(metrics)

        if mode == tf.estimator.ModeKeys.EVAL:
            return create_eval_estimator_spec(mode, metrics, params)

        if mode == tf.estimator.ModeKeys.TRAIN:
            return create_train_estimator_spec(
                mode, nce_weights, nce_biases,
                features, labels, user_vector, params)

    assert False  # Never reach here


def get_nce_variables(params):
    """Get embeddings, nce_weights, nce_biases."""

    opts = params['opts']
    num_classes = params['num_classes']
    embedding_dim = opts.embedding_dim
    train_nce_biases = opts.train_nce_biases

    embeddings = get_embeddings(num_classes, embedding_dim)
    (nce_weights,
     nce_biases) = get_nce_weights_and_biases(
         num_classes, embedding_dim, train_nce_biases)
    return (embeddings, nce_weights, nce_biases)


def get_nce_weights_and_biases(num_classes, embedding_dim, train_nce_biases):
    """Get nce weights and biases variables."""

    with tf.variable_scope("nce_layer", reuse=tf.AUTO_REUSE):
        nce_weights = tf.get_variable(
            model_keys.NCE_WEIGHTS_NAME,
            initializer=tf.truncated_normal(
                [num_classes, embedding_dim],
                stddev=1.0 / math.sqrt(embedding_dim)))
        nce_biases = tf.get_variable(
            model_keys.NCE_BIASES_NAME, initializer=tf.zeros([num_classes]),
            trainable=train_nce_biases)
    return nce_weights, nce_biases


def get_embeddings(num_classes, embedding_dim):
    """Get embeddings variables."""

    with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
        embeddings = tf.get_variable(
            "embeddings",
            initializer=tf.random_uniform([num_classes, embedding_dim],
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


def create_input_layer(mode, features, params, embeddings):
    """Create input layer."""

    feature_columns = params['feature_columns']
    predict_feature_columns = params['predict_feature_columns']
    embedding_dim = params['opts'].embedding_dim

    if (mode == tf.estimator.ModeKeys.PREDICT
            or mode == tf.estimator.ModeKeys.EVAL):
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

    return embeds_mean


def create_hidden_layer(mode, input_layer, params):
    """Create DNN hidden layers."""

    opts = params['opts']
    hidden_units = opts.hidden_units
    dropout = opts.dropout

    hidden = input_layer
    for index, units in enumerate(hidden_units):
        is_last_layer = False
        if index == len(hidden_units) - 1:
            is_last_layer = True

        activation = None if is_last_layer else tf.nn.relu
        dropout = 0.0 if is_last_layer else dropout

        hidden = tf.layers.dense(
            hidden, units=units, activation=activation,
            name="fc{}_{}".format(index, units), reuse=tf.AUTO_REUSE)
        if dropout > 0.0:
            training = (mode == tf.estimator.ModeKeys.TRAIN)
            hidden = tf.layers.dropout(
                hidden, dropout, training=training,
                name="dropout{}_{}".format(index, units))
    return hidden


def get_metrics(labels, ids, params):
    """Get metrics dict."""

    opts = params['opts']
    ntargets = opts.ntargets
    recall_k = opts.recall_k

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


def create_loss(weights, biases, labels, inputs, params):
    """Create nce loss."""

    num_classes = params['num_classes']
    opts = params['opts']
    num_sampled = opts.num_sampled
    ntargets = opts.ntargets

    sampled_values = tf.nn.learned_unigram_candidate_sampler(
        true_classes=labels,
        num_true=ntargets,
        num_sampled=num_sampled,
        unique=True,
        range_max=num_classes,
        seed=np.random.randint(1000000)
    )
    loss = tf.nn.nce_loss(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=inputs,
        num_sampled=num_sampled,
        num_classes=num_classes,
        num_true=ntargets,
        sampled_values=sampled_values,
        partition_strategy="div")
    loss = tf.reduce_mean(loss, name="mean_loss")

    return loss


def create_optimizer(features, params):
    """Create optimizer."""

    global _optimizer_id

    opts = params['opts']
    ntokens = params['ntokens']
    lr = opts.lr
    optimizer_type = opts.optimizer_type

    if optimizer_type == model_keys.OptimizerType.ADA:
        optimizer = tf.train.AdagradOptimizer(
            learning_rate=lr, name='adagrad_{}'.format(_optimizer_id))
    elif optimizer_type == model_keys.OptimizerType.ADADELTA:
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate=lr, name='adadelta_{}'.format(_optimizer_id))
    elif optimizer_type == model_keys.OptimizerType.ADAM:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr, name='adam_{}'.format(_optimizer_id))
    elif optimizer_type == model_keys.OptimizerType.SGD:
        processed_tokens = features[model_keys.TOKENS_COL][0][0]
        tf.summary.scalar("processed_tokens", processed_tokens)
        new_lr = lr * (1.0 - (tf.cast(processed_tokens, tf.float32)
                              / tf.cast(ntokens, tf.float32)))
        new_lr = tf.maximum(new_lr, 1e-5)
        tf.summary.scalar("lr", new_lr)
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=new_lr, name='sgd_{}'.format(_optimizer_id))
    else:
        raise ValueError('OptimizerType "{}" not surpported.'
                         .format(optimizer_type))
    _optimizer_id += 1

    return optimizer


def create_predict_estimator_spec(mode, user_vector, features, params):
    """Create predict EstimatorSpec."""

    opts = params['opts']

    if (opts.optimize_level
            == model_keys.OPTIMIZE_LEVEL_SAVED_NCE_PARAMS):
        tf.logging.info("Use OPTIMIZE_LEVEL_SAVED_NCE_PARAMS")
        scores, ids, _ = optimize_level_saved_nce_params(
            opts.dict_dir, user_vector, opts.recall_k, opts.use_subset)
    elif (opts.optimize_level
          == model_keys.OPTIMIZE_LEVEL_OPENBLAS_TOP_K):
        tf.logging.info("Use OPTIMIZE_LEVEL_OPENBLAS_TOP_K")
        scores, ids = optimize_level_openblas_top_k(
            opts.dict_dir, user_vector, opts.recall_k, opts.use_subset)
    else:
        raise ValueError(
            "not surpported optimize_level '{}'".format(opts.optimize_level))

    table = create_index_to_string_table(opts.dict_dir, opts.use_subset)
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


def create_eval_estimator_spec(mode, metrics, params):
    """Create eval EstimatorSpec."""

    return tf.estimator.EstimatorSpec(
        mode,
        loss=tf.constant(0),  # don't evaluate loss
        eval_metric_ops=metrics)


def create_train_estimator_spec(
        mode, nce_weights, nce_biases, features, labels, user_vector, params):
    """Create train EstimatorSpec."""

    loss = create_loss(nce_weights, nce_biases, labels, user_vector, params)
    optimizer = create_optimizer(features, params)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def compute_top_k(nce_weights, nce_biases, user_vector, params):
    """Compute top k."""

    opts = params['opts']

    logits = tf.nn.xw_plus_b(
        user_vector, tf.transpose(nce_weights), nce_biases)
    scores, ids = tf.nn.top_k(
        logits, opts.recall_k, name="top_k_{}".format(opts.recall_k))
    return scores, ids
