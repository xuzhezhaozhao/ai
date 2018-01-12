#! /usr/bin/env python
# -*-coding:utf-8 -*-

from __future__ import division

import tensorflow as tf
from input_data_binary import load_video_embeddings_from_binary
import math


def model_fn(features, labels, mode, params):
    # loading pretrained word vectors
    video_embeddings_dummy, num_videos, embedding_dim = \
        load_video_embeddings_from_binary(params["embeddings_file_path"])

    embeddings = tf.Variable(
        tf.random_uniform([num_videos, embedding_dim], -1.0, 1.0),
        name="embeddings"
    )
    embed = tf.nn.embedding_lookup(embeddings, features["watched"])
    mean_input = tf.reduce_mean(embed, 1)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([num_videos, embedding_dim],
                            stddev=1.0 / math.sqrt(embedding_dim)))
    nce_biases = tf.Variable(tf.zeros([num_videos]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the meaning of NCE loss:
    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=labels,
                       inputs=mean_input,
                       num_sampled=params["num_sampled"],
                       num_classes=num_videos,
                       name="nce_loss"))

    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
    trainable_variables = tf.trainable_variables()
    gradients = optimizer.compute_gradients(loss, trainable_variables)

    tf.summary.scalar('nce_loss', loss)

    # Add gradients to summary
    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/gradient', gradient)

    train_op = optimizer.apply_gradients(
        grads_and_vars=gradients,
        global_step=tf.train.get_global_step(),
    )

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    # Generate top-K predictions
    probs = tf.matmul(mean_input, normalized_embeddings, transpose_b=True)

    predictions = tf.nn.top_k(probs, params["k"], name="top_k_predictions")
    predictions_k_1 = tf.nn.top_k(probs, 1, name="top_1_predictions")

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                "predicts": predictions.indices,
                "scores": predictions.values
            },
            export_outputs={
                'predicts': tf.estimator.export.PredictOutput(
                    outputs={
                        'predicts': predictions.indices,
                        'scores': predictions.values
                    }
                )
            },
        )

    one_hot_labels = tf.reshape(labels, [-1])

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=one_hot_labels,
            predictions=predictions_k_1.indices
        ),
        "recall_at_k": tf.metrics.recall_at_k(
            labels=tf.cast(one_hot_labels, tf.int64),
            predictions=probs,
            k=params["k"]
        )
    }

    # Provide an estimator spec for 'ModeKeys.EVAL' and 'ModeKeys.TRAIN'
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )
