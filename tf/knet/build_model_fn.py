#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import struct
import tensorflow as tf
import numpy as np

import model_keys
import custom_ops
import utils


_call_model_fn_times = 0


def knet_model_fn(features, labels, mode, params):
    """Build model graph."""

    global _call_model_fn_times
    _call_model_fn_times += 1

    opts = params['opts']

    with tf.name_scope("model_fn_{}".format(_call_model_fn_times)):

        embeddings = get_embeddings(params)
        nce_weights, nce_biases = get_nce_weights_and_biases(params)

        if mode != tf.estimator.ModeKeys.TRAIN and opts.normalize_embeddings:
            embeddings = load_model_embeddings(opts.dict_dir)
            embeddings = tf.convert_to_tensor(embeddings, dtype=tf.float32)

        input_layer = create_input_layer(mode, features, params, embeddings)
        user_vector = create_hidden_layer(mode, input_layer, params)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return create_predict_estimator_spec(
                mode, user_vector, features, params)

        logits, _, top_k_ids = compute_top_k(
            nce_weights, nce_biases, user_vector, params)
        metrics = get_metrics(labels, logits, top_k_ids, params)
        add_metrics_summary(metrics)

        if mode == tf.estimator.ModeKeys.EVAL:
            return create_eval_estimator_spec(
                mode, labels, user_vector, params)

        if mode == tf.estimator.ModeKeys.TRAIN:
            return create_train_estimator_spec(
                mode, nce_weights, nce_biases,
                features, labels, user_vector, params)

    assert False  # Never reach here


def get_embeddings(params):
    """Get embeddings variables."""

    opts = params['opts']
    num_classes = params['num_classes']
    embedding_dim = opts.embedding_dim

    with tf.variable_scope("embeddings_variable", reuse=tf.AUTO_REUSE):
        init_width = 1.0 / embedding_dim
        embeddings = tf.get_variable(
            "embeddings", initializer=tf.random_uniform(
                [num_classes, embedding_dim], -init_width, init_width))
    return embeddings


def get_nce_weights_and_biases(params):
    """Get nce weights and biases variables."""

    opts = params['opts']
    num_classes = params['num_classes']
    user_features_dim = params['user_features_dim']

    if len(opts.hidden_units) == 0:
        nce_dim = opts.embedding_dim
        if opts.use_user_features:
            nce_dim += user_features_dim
    else:
        nce_dim = opts.hidden_units[-1]

    opts.nce_dim = nce_dim

    tf.logging.info("nce_dim = {}".format(nce_dim))

    with tf.variable_scope("nce_layer_variables", reuse=tf.AUTO_REUSE):
        nce_weights = tf.get_variable(
            model_keys.NCE_WEIGHTS_NAME,
            initializer=tf.zeros([num_classes, nce_dim]))
        nce_biases = tf.get_variable(
            model_keys.NCE_BIASES_NAME, initializer=tf.zeros([num_classes]),
            trainable=opts.train_nce_biases)
    return nce_weights, nce_biases


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

    opts = params['opts']
    embedding_dim = opts.embedding_dim
    use_batch_normalization = opts.use_batch_normalization
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    with tf.name_scope("input_layer"):
        if training:
            records_column = params['records_column']
        else:
            records_column = params['predict_records_column']

        records_column = tf.feature_column.input_layer(
            features, records_column)

        # [batch, 1]
        nonzeros = tf.count_nonzero(records_column, 1, keepdims=True)
        nonzeros = tf.maximum(nonzeros, 1)  # avoid divide zero

        if not training and opts.normalize_embeddings:
            embeds = tf.nn.embedding_lookup(
                embeddings, tf.cast(records_column, tf.int32))
        else:
            embeds = mask_padding_embedding_lookup(
                embeddings, embedding_dim, records_column,
                model_keys.PADDING_ID)

        embeds_sum = tf.reduce_sum(embeds, 1)
        embeds_mean = embeds_sum / tf.cast(nonzeros, tf.float32)

        if use_batch_normalization:
            embeds_mean = batch_normalization(
                embeds_mean, training, "bn_input")

        if opts.use_user_features:
            tf.logging.info("Use user features")
            user_features = tf.feature_column.input_layer(
                features, params['user_features_columns'])
            concat_features = [embeds_mean, user_features]
            input_layer = tf.concat(concat_features, axis=1,
                                    name='concat_user_features')
        else:
            input_layer = embeds_mean

    return input_layer


def create_hidden_layer(mode, input_layer, params):
    """Create DNN hidden layers."""

    opts = params['opts']
    hidden_units = opts.hidden_units
    dropout = opts.dropout
    use_batch_normalization = opts.use_batch_normalization
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    with tf.name_scope('hidden_layers'):
        hidden = input_layer
        for index, units in enumerate(hidden_units):
            is_last_layer = False
            if index == len(hidden_units) - 1:
                is_last_layer = True

            activation = None if is_last_layer else tf.nn.relu
            dropout = 0.0 if is_last_layer else dropout

            hidden = tf.layers.dense(
                hidden, units=units, activation=None,
                name="fc{}_{}".format(index, units), reuse=tf.AUTO_REUSE)
            if use_batch_normalization:
                hidden = batch_normalization(
                    hidden, training, "bn{}_{}".format(index, units))
            if activation is not None:
                hidden = activation(hidden)

            if dropout > 0.0:
                hidden = tf.layers.dropout(
                    hidden, dropout, training=training,
                    name="dropout{}_{}".format(index, units))
    return hidden


def get_metrics(labels, logits, ids, params):
    """Get metrics dict."""

    opts = params['opts']
    ntargets = opts.ntargets
    recall_k = opts.recall_k
    recall_k2 = recall_k//2
    recall_k4 = recall_k//4

    with tf.name_scope('eval_metrics'):
        predicted = ids[:, :ntargets]
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted)
        recall_at_top_k = tf.metrics.recall_at_top_k(
            labels=labels, predictions_idx=ids, k=recall_k)
        recall_at_top_k2 = tf.metrics.recall_at_top_k(
            labels=labels, predictions_idx=ids[:, :recall_k2], k=recall_k2)
        recall_at_top_k4 = tf.metrics.recall_at_top_k(
            labels=labels, predictions_idx=ids[:, :recall_k4], k=recall_k4)
        precision_at_top_k = tf.metrics.precision_at_top_k(
            labels=labels, predictions_idx=ids, k=recall_k)
        average_precision_at_k = tf.metrics.average_precision_at_k(
            labels=labels, predictions=logits, k=recall_k)

        metrics = {'accuracy': accuracy,
                   'recall_at_top_{}'.format(recall_k): recall_at_top_k,
                   'recall_at_top_{}'.format(recall_k2): recall_at_top_k2,
                   'recall_at_top_{}'.format(recall_k4): recall_at_top_k4,
                   'precision_at_top_{}'.format(recall_k): precision_at_top_k,
                   'average_precision_at_{}'
                   .format(recall_k): average_precision_at_k}
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
        'nce_layer_variables/' + model_keys.NCE_WEIGHTS_NAME)
    nce_biases = model.get_variable_value(
        'nce_layer_variables/' + model_keys.NCE_BIASES_NAME)
    return (nce_weights, nce_biases)


def get_model_embeddings(model):
    """Get embeddings variables from estimator model"""

    embeddings = model.get_variable_value(
        'embeddings_variable/' + model_keys.EMBEDDINGS_NAME)
    return embeddings


def save_model_nce_params(model, opts):
    """Save model nce weights and biases variables."""

    dict_dir = opts.dict_dir
    nce_weights, nce_biases = get_model_nce_weights_and_biases(model)

    if opts.normalize_nce_weights:
        tf.logging.info("Normalize nce weihts.")
        nce_weights = utils.normalize_matrix(nce_weights)

    tf.logging.info('save nce_weights = \n{}'.format(nce_weights))
    tf.logging.info('save nce_biases = \n{}'.format(nce_biases))
    save_weights_path = os.path.join(
        dict_dir, model_keys.SAVE_NCE_WEIGHTS_NAME)
    save_biases_path = os.path.join(dict_dir, model_keys.SAVE_NCE_BIASES_NAME)
    np.save(save_weights_path, nce_weights)
    np.save(save_biases_path, nce_biases)


def save_model_embeddings(model, opts):
    """Save model embeddings."""

    dict_dir = opts.dict_dir
    embeddings = get_model_embeddings(model)
    if opts.normalize_embeddings:
        tf.logging.info("Normalize embeddings.")
        embeddings = utils.normalize_matrix(embeddings)
    embeddings[model_keys.PADDING_ID] = 0.0
    tf.logging.info('save embeddings = \n{}'.format(embeddings))
    save_embeddings_path = os.path.join(
        dict_dir, model_keys.SAVE_EMBEDDINGS_NAME)
    np.save(save_embeddings_path, embeddings)


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

    return (nce_weights, nce_biases)


def load_model_embeddings(dict_dir):
    """Load model embeddings."""

    embeddings_name = model_keys.SAVE_EMBEDDINGS_NAME
    save_embeddings_path = os.path.join(dict_dir, embeddings_name)
    embeddings = np.load(save_embeddings_path)

    return embeddings


def save_numpy_float_array(array, filename):
    """Save numpy float array to file."""

    with open(filename, 'wb') as f:
        for d in array.shape:
            f.write(struct.pack('<q', d))

        fl = array.flat
        for v in fl:
            f.write(struct.pack('<f', v))


def create_loss(weights, biases, labels, inputs, params):
    """Create nce loss."""

    opts = params['opts']

    with tf.name_scope('loss_layer'):
        # Negative sampling.
        if opts.nce_loss_type == model_keys.NceLossType.DEFAULT:
            tf.logging.info("Use default nce loss.")
            sampled_values = get_negative_samples(labels, params)
            return default_nce_loss(weights, biases, labels, inputs,
                                    sampled_values, params)
        elif opts.nce_loss_type == model_keys.NceLossType.WORD2VEC:
            tf.logging.info("Use word2vec nce loss.")
            sampled_values = get_negative_samples(labels, params)
            return word2vec_nce_loss(weights, biases, labels, inputs,
                                     sampled_values, params)
        elif opts.nce_loss_type == model_keys.NceLossType.FASTTEXT:
            tf.logging.info("Use fasttext nce loss.")
            return fasttext_nce_loss(weights, biases, labels, inputs, params)
        else:
            raise ValueError("Unsurpported nce loss type.")


def word2vec_nce_loss(weights, biases, labels, inputs, sampled_values, params):
    """Optimized custom nce loss implemented.
    ref: tensorflow/models/tutorials/embedding/word2vec.py (github)
    """

    opts = params['opts']

    sampled_ids = sampled_values[0]

    # Weights for labels: [batch_size, emb_dim]
    true_w = tf.nn.embedding_lookup(weights, tf.reshape(labels, [-1]))
    # Biases for labels: [batch_size, emb_dim]
    true_b = tf.nn.embedding_lookup(biases, tf.reshape(labels, [-1]))

    # Weights for sampled ids: [num_sampled, emb_dim]
    sampled_w = tf.nn.embedding_lookup(weights, sampled_ids)
    # Biases for sampled ids: [num_sampled, 1]
    sampled_b = tf.nn.embedding_lookup(biases, sampled_ids)

    # True logits: [batch_size, 1]
    true_logits = tf.reduce_sum(tf.multiply(inputs, true_w), 1) + true_b

    # Sampled logits: [batch_size, num_sampled]
    # We replicate sampled noise labels for all examples in the batch
    # using the matmul.
    sampled_b_vec = tf.reshape(sampled_b, [opts.num_sampled])
    sampled_logits = tf.matmul(inputs,
                               sampled_w,
                               transpose_b=True) + sampled_b_vec

    # cross-entropy(logits, labels)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(true_logits), logits=true_logits)
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

    # NCE-loss is the sum of the true and noise (sampled words)
    # contributions, averaged over the batch.
    nce_loss_tensor = (tf.reduce_sum(true_xent) +
                       tf.reduce_sum(sampled_xent)) / opts.batch_size
    return nce_loss_tensor


def fasttext_nce_loss(weights, biases, labels, inputs, params):
    """Fasttext nce loss. [batch, num_sampled] sampled ids."""

    num_classes = params['num_classes']
    opts = params['opts']

    dict_word_counts_path = os.path.join(
        opts.dict_dir, model_keys.DICT_WORD_COUNTS)
    unigrams = [int(word.strip()) for word in open(dict_word_counts_path)]
    unigrams = tf.make_tensor_proto(unigrams, dtype=tf.int64)

    # [batch, num_sampled]
    sampled_ids = custom_ops.fasttext_negative_sampler(
        true_classes=labels,
        num_true=opts.ntargets,
        num_sampled=opts.num_sampled,
        unique=True,
        range_max=num_classes,
        num_reserved_ids=1,
        unigrams=unigrams,
        seed=np.random.randint(1000000))

    # Weights for labels: [batch_size, emb_dim]
    true_w = tf.nn.embedding_lookup(weights, tf.reshape(labels, [-1]))
    # Biases for labels: [batch_size, emb_dim]
    true_b = tf.nn.embedding_lookup(biases, tf.reshape(labels, [-1]))

    # Weights for sampled ids: [batch, num_sampled, emb_dim]
    sampled_w = tf.nn.embedding_lookup(weights, sampled_ids)
    # Biases for sampled ids: [batch, num_sampled, 1]
    sampled_b = tf.nn.embedding_lookup(biases, sampled_ids)

    # True logits: [batch_size, 1]
    true_logits = tf.reduce_sum(tf.multiply(inputs, true_w), 1) + true_b

    # Sampled logits: [batch_size, num_sampled]
    sampled_b_vec = tf.reshape(sampled_b, [-1, opts.num_sampled])
    broadcast_inputs = tf.reshape(inputs, [-1, 1, opts.nce_dim])
    sampled_logits = tf.multiply(broadcast_inputs, sampled_w)
    sampled_logits = tf.reduce_sum(sampled_logits, -1) + sampled_b_vec

    # cross-entropy(logits, labels)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(true_logits), logits=true_logits)
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

    # NCE-loss is the sum of the true and noise (sampled words)
    # contributions, averaged over the batch.
    nce_loss_tensor = (tf.reduce_sum(true_xent) +
                       tf.reduce_sum(sampled_xent)) / opts.batch_size
    return nce_loss_tensor


def default_nce_loss(weights, biases, labels, inputs, sampled_values, params):
    """Default nce loss."""

    num_classes = params['num_classes']
    opts = params['opts']
    num_sampled = opts.num_sampled
    ntargets = opts.ntargets

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
    loss = tf.reduce_sum(loss, name="mean_loss")
    loss = loss / opts.batch_size
    return loss


def create_optimizer(features, params):
    """Create optimizer."""

    opts = params['opts']
    lr = opts.lr
    optimizer_type = opts.optimizer_type

    with tf.name_scope('optimizer_layer'):
        if optimizer_type == model_keys.OptimizerType.ADA:
            optimizer = tf.train.AdagradOptimizer(
                learning_rate=lr,
                name='adagrad_{}'.format(_call_model_fn_times))
        elif optimizer_type == model_keys.OptimizerType.ADADELTA:
            optimizer = tf.train.AdadeltaOptimizer(
                learning_rate=lr,
                rho=0.95,
                epsilon=0.00001,
                name='adadelta_{}'.format(_call_model_fn_times))
        elif optimizer_type == model_keys.OptimizerType.ADAM:
            optimizer = tf.train.AdamOptimizer(
                learning_rate=lr, name='adam_{}'.format(_call_model_fn_times))
        elif optimizer_type == model_keys.OptimizerType.SGD:
            if (opts.sgd_lr_decay_type
                    == model_keys.SGDLrDecayType.FASTTEXT_DECAY):
                sgd_lr = sgd_lr_fasttext_decay(features, params)
            elif (opts.sgd_lr_decay_type
                  == model_keys.SGDLrDecayType.EXPONENTIAL_DECAY):
                sgd_lr = tf.train.exponential_decay(
                    lr, tf.train.get_global_step(),
                    decay_steps=opts.sgd_lr_decay_steps,
                    decay_rate=opts.sgd_lr_decay_rate)
            elif opts.sgd_lr_decay_type == model_keys.SGDLrDecayType.NONE:
                sgd_lr = lr
            elif (opts.sgd_lr_decay_type
                  == model_keys.SGDLrDecayType.POLYNOMIAL_DECAY):
                sgd_lr = tf.train.polynomial_decay(
                    lr, tf.train.get_global_step(),
                    decay_steps=opts.sgd_lr_decay_steps,
                    end_learning_rate=opts.sgd_lr_decay_end_learning_rate,
                    power=opts.sgd_lr_decay_power,
                    cycle=False)
            else:
                raise ValueError("Unsurpported sgd lr decay type '{}'"
                                 .format(opts.sgd_lr_decay_type))
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=sgd_lr,
                name='sgd_{}'.format(_call_model_fn_times))
            tf.summary.scalar("sgd_lr", sgd_lr)
        elif optimizer_type == model_keys.OptimizerType.RMSPROP:
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=lr,
                decay=0.95,
                momentum=0.001,
                epsilon=1e-10,
                name='rmsprop_{}'.format(_call_model_fn_times))
        else:
            raise ValueError('OptimizerType "{}" not surpported.'
                             .format(optimizer_type))

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


def create_eval_estimator_spec(mode, labels, user_vector, params):
    """Create eval EstimatorSpec."""

    opts = params['opts']
    _, top_k_ids, logits = optimize_level_saved_nce_params(
        opts.dict_dir, user_vector, opts.recall_k, False)
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=tf.reshape(labels, [-1]),
        logits=logits)
    metrics = get_metrics(labels, logits, top_k_ids, params)

    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


def create_train_estimator_spec(
        mode, nce_weights, nce_biases, features, labels, user_vector, params):
    """Create train EstimatorSpec."""

    opts = params['opts']

    loss = create_loss(nce_weights, nce_biases, labels, user_vector, params)
    optimizer = create_optimizer(features, params)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for bn
    with tf.control_dependencies(update_ops):
        if opts.use_clip_gradients:
            gradients, variables = zip(*optimizer.compute_gradients(
                loss, gate_gradients=tf.train.Optimizer.GATE_GRAPH))
            gradients, _ = tf.clip_by_global_norm(gradients, opts.clip_norm)
            train_op = optimizer.apply_gradients(
                zip(gradients, variables),
                global_step=tf.train.get_global_step())
        else:
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step(),
                gate_gradients=tf.train.Optimizer.GATE_OP)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def compute_top_k(nce_weights, nce_biases, user_vector, params):
    """Compute top k."""

    opts = params['opts']

    with tf.name_scope('top_k_layer'):
        logits = tf.nn.xw_plus_b(
            user_vector, tf.transpose(nce_weights), nce_biases)
        scores, ids = tf.nn.top_k(
            logits, opts.recall_k, name="top_k_{}".format(opts.recall_k))
    return logits, scores, ids


def batch_normalization(input, training, name):
    """batch normalization layer."""

    bn = tf.layers.batch_normalization(
        input, axis=1, training=training,
        scale=False, trainable=True,
        name=name, reuse=tf.AUTO_REUSE)
    return bn


def get_processed_tokens():
    """Processed tokens count."""

    with tf.variable_scope("processed_tokens", reuse=tf.AUTO_REUSE):
        processed_tokens = tf.get_variable(
            "processed_tokens", initializer=tf.constant(0.0, dtype=tf.float32))
    return processed_tokens


def sgd_lr_fasttext_decay(features, params):
    """sgd lr fasttext decay."""

    lr = params['opts'].lr
    total_tokens = params['total_tokens']

    processed_tokens = get_processed_tokens()
    current_tokens = tf.reduce_sum(features[model_keys.TOKENS_COL])
    processed_tokens = tf.assign_add(processed_tokens, current_tokens)
    tf.summary.scalar("processed_tokens", processed_tokens)
    new_lr = lr * (1.0 - (tf.cast(processed_tokens, tf.float32)
                          / tf.cast(total_tokens, tf.float32)))
    new_lr = tf.maximum(new_lr, 1e-8)
    return new_lr


def get_fixed_unigram_negative_samples(labels, params):
    """Get fixed_unigram_candidate_sampler."""

    num_classes = params['num_classes']
    opts = params['opts']

    dict_word_counts_path = os.path.join(
        opts.dict_dir, model_keys.DICT_WORD_COUNTS)
    vocab_counts = [int(word.strip()) for word in open(dict_word_counts_path)]
    vocab_counts.insert(0, vocab_counts[0])  # for padding id
    sampled_values = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels,
        num_true=opts.ntargets,
        num_sampled=opts.num_sampled,
        unique=True,
        range_max=num_classes,
        distortion=0.75,
        # num_reserved_ids=1,  # don't use it seems better?
        unigrams=vocab_counts))
    return sampled_values


def get_log_uniform_negative_samples(labels, params):
    """Get log_uniform_candidate_sampler."""

    num_classes = params['num_classes']
    opts = params['opts']
    sampled_values = (tf.nn.log_uniform_candidate_sampler(
        true_classes=labels,
        num_true=opts.ntargets,
        num_sampled=opts.num_sampled,
        unique=True,
        range_max=num_classes))
    return sampled_values


def get_negative_samples(labels, params):
    """Get negative samples."""

    opts = params['opts']

    if opts.negative_sampler_type == model_keys.NegativeSamplerType.FIXED:
        tf.logging.info("Use fixed_unigram_candidate_sampler.")
        sampled_values = get_fixed_unigram_negative_samples(labels, params)
    elif (opts.negative_sampler_type
          == model_keys.NegativeSamplerType.LOG_UNIFORM):
        tf.logging.info("Use log_uniform_candidate_sampler.")
        sampled_values = get_log_uniform_negative_samples(labels, params)
    else:
        raise ValueError("Unsurpported negative sampler type.")
    return sampled_values
