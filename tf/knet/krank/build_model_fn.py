#!/ usr / bin / env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model_keys


_call_model_fn_times = 0


def clear_model_fn_times():
    global _call_model_fn_times
    _call_model_fn_times = 0


def krank_model_fn(features, labels, mode, params):
    """Build model graph."""

    global _call_model_fn_times
    _call_model_fn_times += 1

    opts = params['opts']
    rowkey_embedding_dim = opts.rowkey_embedding_dim

    rowkey_embeddings = get_rowkey_embeddings(params)
    positive_records = features[model_keys.POSITIVE_RECORDS_COL]
    negative_records = features[model_keys.NEGATIVE_RECORDS_COL]
    targets = features[model_keys.TARGETS_COL]

    positive_records.set_shape([None, opts.train_ws])
    negative_records.set_shape([None, opts.train_ws])
    targets.set_shape([None])

    positive_embeds_mean = non_zero_mean(rowkey_embeddings,
                                         rowkey_embedding_dim,
                                         positive_records)
    negative_embeds_mean = non_zero_mean(rowkey_embeddings,
                                         rowkey_embedding_dim,
                                         positive_records)
    targets_embeds = mask_padding_embedding_lookup(
        rowkey_embeddings, rowkey_embedding_dim, targets, 0)

    concat_features = [positive_embeds_mean,
                       negative_embeds_mean,
                       targets_embeds]

    hidden = tf.concat(concat_features, 1)
    num_in = hidden.shape[-1].value
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    # TODO
    hidden = batch_normalization(hidden, training, 'bn_input')
    for index, units in enumerate(opts.hidden_units):
        use_relu = False if index == (len(opts.hidden_units) - 1) else True
        hidden = fc(params, hidden, num_in, units, "fc_{}".format(index),
                    bn=use_relu, relu=use_relu, training=training,
                    dropout=opts.dropout)
        num_in = units

    lr_dim = hidden.shape[-1].value
    lr_weights, lr_biases = get_lr_weights_and_biases(params, lr_dim)
    logits = tf.reduce_sum(hidden * lr_weights, 1) + lr_biases

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

    ce_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(labels, tf.float32), logits=logits))
    l2_loss = tf.losses.get_regularization_loss()
    loss = ce_loss + l2_loss
    tf.summary.scalar('ce_loss', ce_loss)
    tf.summary.scalar('l2_loss', l2_loss)
    tf.summary.scalar('total_loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = create_optimizer(params)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for bn
        with tf.control_dependencies(update_ops):
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            if opts.clip_gradients:
                clip_value = opts.clip_gradients_norm
                gradients = [
                    None if gradient is None else tf.clip_by_value(
                        gradient, -clip_value, clip_value)
                    for gradient in gradients]
            train_op = optimizer.apply_gradients(
                zip(gradients, variables),
                global_step=tf.train.get_global_step())

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
                             num_thresholds=opts.auc_num_thresholds)

        metrics = {
            'accuracy': accuracy,
            'auc': auc,
        }

        return tf.estimator.EstimatorSpec(mode, loss=loss,
                                          eval_metric_ops=metrics)


def l2_regularizer(params):
    """Return L2 regularizer."""

    opts = params['opts']
    if opts.l2_regularizer > 0.0:
        return tf.contrib.layers.l2_regularizer(scale=opts.l2_regularizer)
    else:
        return None


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
        tf.summary.histogram("embeddings", embeddings)
    return embeddings


def non_zero_mean(embeddings, dim, records):
    nonzeros = tf.count_nonzero(records, 1, keepdims=True)
    nonzeros = tf.maximum(nonzeros, 1)
    embeds = mask_padding_embedding_lookup(embeddings, dim, records, 0)
    embeds_sum = tf.reduce_sum(embeds, 1)
    embeds_mean = embeds_sum / tf.cast(nonzeros, tf.float32)
    return embeds_mean


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


def batch_normalization(input, training, name='bn'):
    """batch normalization layer."""

    bn = tf.layers.batch_normalization(
        input, axis=1, training=training,
        scale=False, trainable=True,
        name=name, reuse=tf.AUTO_REUSE)
    return bn


def get_lr_weights_and_biases(params, dim):
    """Get logistic regression weights and biases."""

    with tf.variable_scope('logiistic_regression', reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(
            'weights',
            initializer=tf.random_uniform([dim], -0.1, 0.1),
            regularizer=l2_regularizer(params))
        biases = tf.get_variable('biases', initializer=[0.0], dtype=tf.float32)

        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
    return weights, biases


def fc(params, x, num_in, num_out, name, bn=True, relu=True,
       training=True, dropout=0.0):
    """Create a fully connected layer."""

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        w = tf.sqrt(12.0 / (num_in + num_out))
        weights = tf.get_variable(
            'weights',
            initializer=tf.random_uniform([num_in, num_out], -w, w),
            regularizer=l2_regularizer(params))
        biases = tf.get_variable(
            'biases',
            initializer=tf.zeros([num_out]),
            regularizer=l2_regularizer(params))
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if bn:
            act = batch_normalization(act, training)

        if relu:
            act = tf.nn.leaky_relu(act)

        if dropout > 0.0:
            act = tf.layers.dropout(act, dropout, training=training)

        return act


def create_optimizer(params):
    """Create optimizer."""

    opts = params['opts']
    optimizer_type = opts.optimizer_type

    # used for GradientDescentOptimizer and MomentumOptimizer
    decay_lr = tf.train.exponential_decay(
        learning_rate=opts.lr,
        global_step=tf.train.get_global_step(),
        decay_steps=opts.optimizer_exponential_decay_steps,
        decay_rate=opts.optimizer_exponential_decay_rate,
        staircase=opts.optimizer_exponential_decay_staircase)

    with tf.name_scope('optimizer_layer'):
        if optimizer_type == model_keys.OptimizerType.ADA:
            optimizer = tf.train.AdagradOptimizer(
                learning_rate=opts.lr,
                name='adagrad_{}'.format(_call_model_fn_times))
        elif optimizer_type == model_keys.OptimizerType.ADADELTA:
            optimizer = tf.train.AdadeltaOptimizer(
                learning_rate=opts.lr,
                rho=opts.optimizer_adadelta_rho,
                epsilon=opts.optimizer_epsilon,
                name='adadelta_{}'.format(_call_model_fn_times))
        elif optimizer_type == model_keys.OptimizerType.ADAM:
            optimizer = tf.train.AdamOptimizer(
                learning_rate=opts.lr,
                beta1=opts.optimizer_adam_beta1,
                beta2=opts.optimizer_adam_beta2,
                epsilon=opts.optimizer_epsilon,
                name='adam_{}'.format(_call_model_fn_times))
        elif optimizer_type == model_keys.OptimizerType.SGD:
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=decay_lr,
                name='sgd_{}'.format(_call_model_fn_times))
        elif optimizer_type == model_keys.OptimizerType.RMSPROP:
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=opts.lr,
                decay=opts.optimizer_rmsprop_decay,
                momentum=opts.optimizer_rmsprop_momentum,
                epsilon=opts.optimizer_epsilon,
                centered=opts.optimizer_rmsprop_centered,
                name='rmsprop_{}'.format(_call_model_fn_times))
        elif optimizer_type == model_keys.OptimizerType.MOMENTUM:
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=decay_lr,
                momentum=opts.optimizer_momentum_momentum,
                use_nesterov=opts.optimizer_momentum_use_nesterov,
                name='momentum_{}'.format(_call_model_fn_times))
        elif optimizer_type == model_keys.OptimizerType.FTRL:
            optimizer = tf.train.FtrlOptimizer(
                learning_rate=opts.lr,
                learning_rate_power=opts.optimizer_ftrl_lr_power,
                initial_accumulator_value=opts.optimizer_ftrl_initial_accumulator_value,
                l1_regularization_strength=opts.optimizer_ftrl_l1_regularization,
                l2_regularization_strength=opts.optimizer_ftrl_l2_regularization,
                name='ftrl_{}'.format(_call_model_fn_times),
                l2_shrinkage_regularization_strength=opts.optimizer_ftrl_l2_shrinkage_regularization)
        else:
            raise ValueError('OptimizerType "{}" not surpported.'
                             .format(optimizer_type))

    return optimizer
