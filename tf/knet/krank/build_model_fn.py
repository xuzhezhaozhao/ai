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

    if opts.target_use_share_embeddings:
        target_rowkey_embeddings = rowkey_embeddings
        targte_rowkey_embedding_dim = rowkey_embedding_dim
    else:
        target_rowkey_embeddings = get_target_rowkey_embeddings(params)
        targte_rowkey_embedding_dim = opts.target_rowkey_embedding_dim

    positive_records = features[model_keys.POSITIVE_RECORDS_COL]
    negative_records = features[model_keys.NEGATIVE_RECORDS_COL]
    targets = features[model_keys.TARGETS_COL]
    first_videos = features[model_keys.FIRST_VIDEOS_COL]

    positive_records.set_shape([None, opts.train_ws])
    negative_records.set_shape([None, opts.train_ws])
    targets.set_shape([None])
    first_videos.set_shape([None])

    positive_embeds_mean = non_zero_mean(rowkey_embeddings,
                                         rowkey_embedding_dim,
                                         positive_records)
    negative_embeds_mean = non_zero_mean(rowkey_embeddings,
                                         rowkey_embedding_dim,
                                         negative_records)
    targets_embeds = mask_padding_embedding_lookup(
        target_rowkey_embeddings, targte_rowkey_embedding_dim, targets, 0)
    first_videos_embeds = mask_padding_embedding_lookup(
        rowkey_embeddings, rowkey_embedding_dim, first_videos, 0)

    tf.summary.histogram('positive_embeds_mean', positive_embeds_mean)
    tf.summary.histogram('negative_embeds_mean', negative_embeds_mean)
    tf.summary.histogram('targets_embeds', targets_embeds)
    tf.summary.histogram('first_videos_embeds', first_videos_embeds)

    concat_features = [
        # positive_embeds_mean,
        # negative_embeds_mean,
        first_videos_embeds,
        targets_embeds
    ]

    hidden = tf.concat(concat_features, 1)
    num_in = hidden.shape[-1].value
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    hidden = batch_normalization(hidden, training, 'bn_input')
    for index, units in enumerate(opts.hidden_units):
        use_relu = False if index == (len(opts.hidden_units) - 1) else True
        hidden = fc(params, hidden, num_in, units, "fc_{}".format(index),
                    bn=use_relu, relu=use_relu, training=training,
                    dropout=opts.dropout)
        num_in = units

    if opts.target_skip_connection:
        hidden = tf.concat([hidden, targets_embeds], 1)

    lr_dim = hidden.shape[-1].value
    lr_weights, lr_biases = get_lr_weights_and_biases(params, lr_dim)
    # logits = tf.reduce_sum(hidden * lr_weights, 1) + lr_biases
    logits = tf.reshape(tf.matmul(hidden, lr_weights) + lr_biases, [-1])

    # scores = tf.clip_by_value(tf.nn.sigmoid(logits), 1e-6, 1-1e-6)
    scores = tf.nn.sigmoid(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        is_target_in_dict = features[model_keys.IS_TARGET_IN_DICT_COL]
        num_positive = features[model_keys.NUM_POSITIVE_COL]
        num_negative = features[model_keys.NUM_NEGATIVE_COl]
        retdict = {
            'scores': scores,
            'rowkeys': features[model_keys.TARGET_ROWKEYS_COL],
            model_keys.IS_TARGET_IN_DICT_COL: is_target_in_dict,
            model_keys.NUM_POSITIVE_COL: num_positive,
            model_keys.NUM_NEGATIVE_COl: num_negative,
        }
        predictions = retdict
        export_outputs = {
            'predicts': tf.estimator.export.PredictOutput(outputs=retdict)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)

    tf.summary.histogram('labels', labels)
    tf.summary.histogram('scores', scores)

    target_labels = labels
    binary_labels = tf.to_int32(labels >= opts.binary_label_threhold)
    if opts.use_binary_label:
        target_labels = binary_labels
        tf.summary.histogram('binary_labels', binary_labels)

    if opts.loss_type == model_keys.LossType.CE:
        m_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(target_labels, tf.float32), logits=logits))
    elif opts.loss_type == model_keys.LossType.MSE:
        m_loss = tf.reduce_mean(tf.losses.mean_squared_error(
            labels=tf.cast(target_labels, tf.float32), predictions=scores))
    else:
        raise ValueError("Unsurpported loss type.")

    l2_loss = tf.losses.get_regularization_loss()
    loss = m_loss + l2_loss
    tf.summary.scalar('m_loss', m_loss)
    tf.summary.scalar('l2_loss', l2_loss)
    tf.summary.scalar('total_loss', loss)

    bool_scores = tf.to_int32(scores > 0.5)
    accuracy = tf.metrics.accuracy(labels=binary_labels,
                                   predictions=bool_scores)
    mse = tf.metrics.mean_squared_error(labels=labels,
                                        predictions=scores)
    auc = tf.metrics.auc(labels=binary_labels,
                         predictions=scores,
                         num_thresholds=opts.auc_num_thresholds)
    tf.summary.scalar('train_accuracy', tf.reduce_mean(accuracy))
    tf.summary.scalar('train_auc', tf.reduce_mean(auc))

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = create_optimizer(params)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for bn
        if opts.use_variable_averages:
            variable_averages = tf.train.ExponentialMovingAverage(
                opts.moving_avg_decay, tf.train.get_global_step())
            variable_averages_op = variable_averages.apply(
                 [v for v in tf.trainable_variables()
                  if v.name.find('embedding') < 0])
            update_ops.append(variable_averages_op)

        with tf.control_dependencies(update_ops):
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            if opts.clip_gradients:
                gradients, _ = tf.clip_by_global_norm(
                    gradients, opts.clip_gradients_norm)
            train_op = optimizer.apply_gradients(
                zip(gradients, variables),
                global_step=tf.train.get_global_step())

            for var, grad in zip(variables, gradients):
                tf.summary.histogram(var.name.replace(':', '_') + '/gradient',
                                     grad)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
            'accuracy': accuracy,
            'auc': auc,
            'mse': mse,
        }

        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)


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
        init_width = 0.5 / dim
        embeddings = tf.get_variable(
            "embeddings",
            initializer=tf.random_uniform([num_rowkey, dim],
                                          -init_width, init_width))
        tf.summary.histogram("embeddings", embeddings)
    return embeddings


def get_target_rowkey_embeddings(params):
    """Get target embeddings variables."""

    opts = params['opts']
    num_rowkey = opts.num_rowkey
    dim = opts.target_rowkey_embedding_dim

    with tf.variable_scope("target_rowkey_embeddings", reuse=tf.AUTO_REUSE):
        init_width = 0.5 / dim
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


def mask_padding_embedding_lookup(embeddings, dim, input, padding_id):
    """ mask padding tf.nn.embedding_lookup.
    padding_id must be zero.

    ref(@ay27): https://github.com/tensorflow/tensorflow/issues/2373
    """

    assert padding_id == 0

    mask_padding_zero_op = tf.scatter_update(
        embeddings, padding_id, tf.zeros([dim], dtype=tf.float32),
        name="mask_padding_zero_op")
    with tf.control_dependencies([mask_padding_zero_op]):
        output = tf.nn.embedding_lookup(embeddings, input)
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
        init_width = 1.0 / dim
        weights = tf.get_variable(
            'weights',
            initializer=tf.random_uniform([dim, 1], -init_width, init_width),
            regularizer=l2_regularizer(params))
        biases = tf.get_variable('biases', initializer=[0.0], dtype=tf.float32)

        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
    return weights, biases


def fc(params, x, num_in, num_out, name, bn=True, relu=True,
       training=True, dropout=0.0):
    """Create a fully connected layer."""

    opts = params['opts']

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
            if opts.leaky_relu_alpha > 0.0:
                act = tf.nn.leaky_relu(act, alpha=opts.leaky_relu_alpha)
            else:
                act = tf.nn.relu(act)

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
    tf.summary.scalar("decay_lr", decay_lr)

    with tf.name_scope('optimizer_layer'):
        if optimizer_type == model_keys.OptimizerType.ADAGRAD:
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
            l1 = opts.optimizer_ftrl_l1_regularization
            l2 = opts.optimizer_ftrl_l2_regularization
            init_value = opts.optimizer_ftrl_initial_accumulator_value
            l2_shrinkage = opts.optimizer_ftrl_l2_shrinkage_regularization
            optimizer = tf.train.FtrlOptimizer(
                learning_rate=opts.lr,
                learning_rate_power=opts.optimizer_ftrl_lr_power,
                initial_accumulator_value=init_value,
                l1_regularization_strength=l1,
                l2_regularization_strength=l2,
                name='ftrl_{}'.format(_call_model_fn_times),
                l2_shrinkage_regularization_strength=l2_shrinkage)
        else:
            raise ValueError('OptimizerType "{}" not surpported.'
                             .format(optimizer_type))

    return optimizer
