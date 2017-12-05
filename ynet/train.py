#! /usr/bin/env python
#-*-coding:utf-8 -*-

from ynet import YNet
import tensorflow as tf
import numpy as np
import argparse
import sys
import os

# Basic model parameters as external flags.
FLAGS = None


def nce_loss(embeddings,
             biases,
             predicts,
             user_vectors,
             num_classes,
             name="nce_loss"):
    """Computes and returns the noise-contrastive estimation training loss.

    Args:
        uv: A Tensor of shape [batch_size, dim]. The forward activations of
        the YNet.
        labels: A Tensor of type int64 and shape [batch_size].
        The target classes.
        embeddings: A Tensor of shape [num_classes, dim]
        biases: A Tensor of shape [num_classes]. The class biases.
        num_classes: An int. The number of target classes
        name:

    Return:
        loss: nce loss
    """
    losses = tf.nn.nce_loss(
        weights=embeddings,
        biases=biases,
        labels=predicts,
        inputs=user_vectors,
        num_sampled=FLAGS.num_sampled,
        num_classes=num_classes,
        name=name
    )
    return tf.reduce_mean(losses, name=name + '_mean')


def training(loss, lr):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable
    variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
        loss: Loss tensor, from nce_loss()
        lr: The learning rate to use for gradient descent.

    Returns:
        train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # optimizer = tf.train.GradientDescentOptimizer(lr)
    optimizer = tf.train.AdamOptimizer(lr)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training
    # step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def run_training():
    """Train YNet for a number of epoches."""
    batch_size = FLAGS.batch_size
    keep_prob = FLAGS.keep_prob
    learning_rate = FLAGS.learning_rate
    epoches = FLAGS.epoches
    history_size = FLAGS.history_size

    with tf.Graph().as_default():
        # 加载训练好的词向量
        video_embeddings, num_videos, embedding_dim, D = \
            load_video_embeddings()
        video_biases = tf.Variable(tf.zeros([num_videos]))

        # 用户浏览历史 placeholder
        histories_placeholder = tf.placeholder(tf.int32,
                                               shape=(batch_size,
                                                      history_size))
        # 待预测的视频 placeholder
        predicts_placeholder = tf.placeholder(tf.int32, shape=(batch_size, 1))

        inputs = generate_average_inputs(video_embeddings,
                                         histories_placeholder)

        # 构造 DNN 网络
        model = YNet(inputs, keep_prob, embedding_dim)
        user_vectors = model.user_vectors

        # 负采样算法
        loss = nce_loss(video_embeddings,
                        video_biases,
                        predicts_placeholder,
                        user_vectors,
                        num_videos)
        train_op = training(loss, learning_rate)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)


def generate_average_inputs(video_embeddings, histories_placeholder):
    """Generate average tensor of embedded video watches.

    Args:
        video_embeddings: Video embeddings.
        histories_placeholder: A shape (batch, history_size) Tensor.
        Placeholder of user watch histories

    Return:
        mean: Average tensor of embedded video watches
    """
    x = tf.gather(video_embeddings, histories_placeholder)
    mean = tf.reduce_mean(x, 1)
    return mean


def load_video_embeddings():
    """ Load pretrained video embeddings from file
    Return:
        embeddings: A shape (num, dim) Tensor. Pretrained video embeddings.
        num: A int. Number of videos.
        dim: A int. Embedding dimension.
        D: A python dict. video key - embedding index in embeddings.
    """
    num = 0
    dim = 0
    D = dict()

    filename = FLAGS.video_embeddings_file
    with open(filename, "r") as f:
        line = f.readline().strip()
        tokens = line.split(' ')
        num, dim = map(int, tokens)

    embeddings = np.genfromtxt(filename, dtype='float32', delimiter=' ',
                               skip_header=2, usecols=range(1, dim + 1))
    embeddings = tf.convert_to_tensor(embeddings, dtype='float32')

    keys = np.genfromtxt(filename, dtype='string', delimiter=' ',
                         skip_header=2, usecols=0)
    D = {key: index for index, key in enumerate(keys)}
    return embeddings, num, dim, D


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )

    parser.add_argument(
        '--epoches',
        type=int,
        default=20,
        help='Number of epoches to run trainer.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )

    parser.add_argument(
        '--keep_prob',
        type=float,
        default=0.5,
        help='dropout keep probability.'
    )

    parser.add_argument(
        '--input_size',
        type=int,
        default=256,
        help='Embedding input size.'
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'logs/ynet'),
        help='Directory to put the log data.'
    )

    parser.add_argument(
        '--num_sampled',
        type=int,
        default=10,
        help='number of sampled of NCE loss.'
    )

    parser.add_argument(
        '--history_size',
        type=int,
        default=50,
        help="User's input history size."
    )

    parser.add_argument(
        '--video_embeddings_file',
        type=str,
        default='video_embeddings.vec',
        help='Pretrained video embeddings file.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
