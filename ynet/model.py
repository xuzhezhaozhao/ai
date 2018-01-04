#! /usr/bin/env python
# -*-coding:utf-8 -*-

from __future__ import division

import tensorflow as tf
from input_data_binary import load_video_embeddings_from_binary


def model_fn(features, labels, mode, params):
    # loading pretrained word vectors
    video_embeddings_dummy, num_videos, embedding_dim = \
        load_video_embeddings_from_binary(params["embeddings_file_path"])

    video_embeddings = tf.Variable(
        tf.random_uniform(shape=[num_videos, embedding_dim], minval=0,
                          maxval=1.0/num_videos),
        trainable=True,
        name="video_embeddings"
    )

    video_biases = tf.Variable(
        tf.zeros([num_videos]),
        trainable=True,
        name="video_biases"
    )

    x = tf.gather(video_embeddings, features["watched"])
    mean_input = tf.reduce_mean(x, 1)

    # Generate top-K predictions
    probs = tf.nn.bias_add(
        tf.matmul(mean_input, video_embeddings, transpose_b=True),
        video_biases,
        name="probs"
    )

    predictions = tf.nn.top_k(probs, params["k"], name="top_k_predictions")
    predictions_k_1 = tf.nn.top_k(probs, 1, name="top_1_predictions")

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                "predicts": predictions.indices,
                # "scores": tf.exp(predictions.values),
                "scores": predictions.values,
                "probs": probs
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

    # Calculate loss
    losses = tf.nn.nce_loss(
        weights=video_embeddings,
        biases=video_biases,
        labels=labels,
        inputs=mean_input,
        num_sampled=params["num_sampled"],
        num_classes=num_videos,
        remove_accidental_hits=True,
        name="nce_losses"
    )
    loss = tf.reduce_mean(losses, name="nce_loss_mean")

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
