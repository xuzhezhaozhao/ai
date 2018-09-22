#!/ usr / bin / env python
#- * - coding = utf8 - * -

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

    hidden = tf.concat(concat_features, 1)
    num_in = hidden.shape[-1].value
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    # TODO
    hidden = batch_normalization(hidden, training, 'bn_input')
    for index, units in enumerate(opts.hidden_units):
        use_relu = False if index == (len(opts.hidden_units) - 1) else True
        hidden = fc(hidden, num_in, units, "fc_{}".format(index),
                    bn=use_relu, relu=use_relu, training=training,
                    dropout=opts.dropout)
        num_in = units

    lr_dim = hidden.shape[-1].value
    lr_weights, lr_biases = get_lr_weights_and_biases(params, lr_dim)

    logits = tf.reduce_sum(hidden * lr_weights, 1)
    logits = logits + lr_biases

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

    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(labels, tf.float32), logits=logits)
    loss = tf.reduce_mean(loss)

    global_step = tf.train.get_global_step()
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=opts.lr)

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
        weights = tf.get_variable('weights',
                                  initializer=tf.random_uniform([dim],
                                                                -0.1, 0.1))
        biases = tf.get_variable('biases', initializer=[0.5], dtype=tf.float32)

        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
    return weights, biases


def fc(x, num_in, num_out, name, bn=True, relu=True,
       training=True, dropout=0.0):
    """Create a fully connected layer."""

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        w = tf.sqrt(12.0 / (num_in + num_out))
        weights = tf.get_variable('weights',
                                  initializer=tf.random_uniform(
                                      [num_in, num_out], -w, w))
        biases = tf.get_variable('biases',
                                 initializer=tf.zeros([num_out]))
#Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if bn:
            act = batch_normalization(act, training)

        if relu:
#Apply ReLu non linearity
#act = tf.nn.relu(act)
            act = tf.nn.leaky_relu(act)

        if dropout > 0.0:
            act = tf.layers.dropout(act, dropout, training=training)

        return act
