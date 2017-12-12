#! /usr/bin/env python
# -*-coding:utf-8 -*-

from __future__ import division

import tensorflow as tf
from input_data_binary import load_video_embeddings_from_binary
import numpy as np


def model_fn(features, labels, mode, params):
    # loading pretrained word vectors
    video_embeddings, num_videos, embedding_dim = \
        load_video_embeddings_from_binary(params["embeddings_file_path"])
    video_embeddings = tf.Variable(
        video_embeddings,
        trainable=params["embeddings_trainable"],
        name="video_embeddings"
    )
    video_biases = tf.Variable(
        tf.zeros([num_videos]),
        # trainable=params["embeddings_trainable"],
        trainable=False,
        name="video_biases"
    )

    x = tf.gather(video_embeddings, features["watched"])
    mean_input = tf.reduce_mean(x, 1)

    keep_prob = params["keep_prob"]
    """Model function for Estimator."""
    # Connect the first hedden layer to input layer
    # (features["watched"]) with relu activation
    num_first = 512
    first_hidden_layer = tf.layers.dense(
        inputs=mean_input,
        units=num_first,
        activation=tf.nn.relu6,
        kernel_initializer=tf.truncated_normal_initializer(
            0, np.sqrt(1.0/num_first)),
        kernel_regularizer=tf.contrib.layers.l1_regularizer(0.01),
        name='fc1'
    )
    first_hidden_layer = tf.nn.dropout(first_hidden_layer, keep_prob)

    loss_parm = params["loss"]
    if loss_parm == "nce":
        num_output = params["embedding_dim"]
    elif loss_parm == "softmax":
        num_output = num_videos
    else:
        raise Exception("Loss function not supported.")

    output_layer = tf.layers.dense(
        inputs=first_hidden_layer,
        units=num_output,
        # activation=tf.nn.relu6,
        activation=tf.nn.sigmoid,
        kernel_initializer=tf.truncated_normal_initializer(
            0, np.sqrt(1.0/num_output)),
        kernel_regularizer=tf.contrib.layers.l1_regularizer(0.01),
        name='fc2'
    )

    # Generate top-K predictions
    if loss_parm == "nce":
        probs = tf.nn.bias_add(
            tf.matmul(output_layer, video_embeddings, transpose_b=True),
            video_biases,
            name="probs"
        )
    elif loss_parm == "softmax":
        probs = tf.exp(output_layer)
    else:
        raise Exception("Loss function not supported.")

    predictions = tf.nn.top_k(probs, params["k"], name="top_k_predictions")
    predictions_k_1 = tf.nn.top_k(probs, 1, name="top_1_predictions")

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                "indices": predictions.indices,
                "values": tf.exp(predictions.values)
            },
            export_outputs={
                # TODO indices to rowkey
                'class': tf.estimator.export.ClassificationOutput(
                    classes=tf.as_string(predictions.indices),
                    scores=tf.exp(predictions.values)),
            },
        )

    one_hot_labels = tf.reshape(labels, [-1])

    # Calculate loss
    if loss_parm == 'nce':
        losses = tf.nn.nce_loss(
            weights=video_embeddings,
            biases=video_biases,
            labels=labels,
            inputs=output_layer,
            num_sampled=params["num_sampled"],
            num_classes=num_videos,
            remove_accidental_hits=True,
            name="nce_losses"
        )
        loss = tf.reduce_mean(losses, name="nce_loss_mean")
    elif loss_parm == 'softmax':
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=one_hot_labels,
            logits=output_layer
        )
        loss = tf.reduce_mean(losses, name="cross_entropy_loss_mean")
    else:
        raise Exception("Loss function not supported.")

    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
    trainable_variables = tf.trainable_variables()
    gradients = optimizer.compute_gradients(loss, trainable_variables)

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
