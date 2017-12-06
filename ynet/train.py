#! /usr/bin/env python
# -*-coding:utf-8 -*-

from ynet import YNet
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base
import numpy as np
import argparse
import sys
import os

# Basic model parameters as external flags.
FLAGS = None

# dict for video key to embeddings index
D = None


class DataSet(object):
    def __init__(self, records):
        """Construct a DataSet."""
        self._records = records
        self._num_examples = records.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def records(self):
        return self._records

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._records = [self.records[i] for i in perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            records_rest_part, predicts_rest_part = \
                generate_batch(self._records[start:self._num_examples])
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._records = [self.records[i] for i in perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            records_new_part, predicts_new_part = \
                generate_batch(self._records[start:end])
            return np.concatenate((records_rest_part, predicts_rest_part), axis=0), np.concatenate((predicts_rest_part, predicts_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return generate_batch(self._records[start:end])


def generate_batch(records):
    watched = []
    predicts = []
    watched_size = FLAGS.watched_size
    for record in records:
        pass
    return watched, predicts


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
    watched_size = FLAGS.watched_size

    with tf.Graph().as_default():
        # 加载训练好的词向量
        video_embeddings, num_videos, embedding_dim, D = \
            load_video_embeddings()
        video_biases = tf.Variable(tf.zeros([num_videos]))

        # 用户浏览历史 placeholder
        watched_pl = tf.placeholder(tf.int32, shape=(batch_size, watched_size))
        # 待预测的视频 placeholder
        predicts_pl = tf.placeholder(tf.int32, shape=(batch_size, 1))

        inputs = generate_average_inputs(video_embeddings, watched_pl)

        # 构造 DNN 网络
        model = YNet(inputs, keep_prob, embedding_dim)
        user_vectors = model.user_vectors

        # 负采样算法
        loss = nce_loss(video_embeddings,
                        video_biases,
                        predicts_pl,
                        user_vectors,
                        num_videos)
        train_op = training(loss, learning_rate)
        init = tf.global_variables_initializer()

        data_sets = read_data_sets(FLAGS.train_file,
                                   FLAGS.validation_file,
                                   FLAGS.test_file)
        with tf.Session() as sess:
            sess.run(init)
            for step in xrange(epoches):
                feed_dict = fill_feed_dict(data_sets.train,
                                           watched_pl,
                                           predicts_pl)
                # Run one step of the model.  The return values are the
                # activations from the `train_op` (which is discarded) and
                # the `loss` Op.  To inspect the values of your Ops or
                # variables, you may include them in the list passed to
                # sess.run() and the value tensors will be returned in the
                # tuple from the call.
                _, loss_value = sess.run([train_op, loss],
                                         feed_dict=feed_dict)
                print("loss: {}".format(loss_value))


def generate_average_inputs(video_embeddings, watched_pl):
    """Generate average tensor of embedded video watches.

    Args:
        video_embeddings: A shape (num_videos, dim) Tensor. Video embeddings.
        watched_pl: A shape (batch, watched_size) Tensor.
        Placeholder of user watched histories

    Return:
        mean: Average tensor of embedded video watches
    """
    x = tf.gather(video_embeddings, watched_pl)
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
    global D

    num = 0
    dim = 0

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
    return embeddings, num, dim


def fill_feed_dict(data_set, watched_pl, predicts_pl):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
    }

    Args:
        data_set:
        watched_pl:
        predicts_pl:

    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    watched_feed, predicts_feed = data_set.next_batch(FLAGS.batch_size)
    feed_dict = {
        watched_pl: watched_feed,
        predicts_pl: predicts_feed,
    }
    return feed_dict


def read_data_sets(train_file, validation_file, test_file):
    global D

    train_records = []
    for line in open(train_file):
        items = line.split(' ')[1:]
        record = [D[k] for k in items]
        train_records.append(record)

    validation_records = []
    for line in open(validation_file):
        items = line.split(' ')[1:]
        record = [D[k] for k in items]
        validation_records.append(record)

    test_records = []
    for line in open(test_file):
        items = line.split(' ')[1:]
        record = [D[k] for k in items]
        test_records.append(record)

    train = DataSet(train_records)
    validation = DataSet(validation_records)
    test = DataSet(test_records)

    return base.Datasets(train=train, validation=validation, test=test)


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
        '--max_steps',
        type=int,
        default=20000,
        help='Number of max steps to run trainer.'
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
        '--watched_size',
        type=int,
        default=50,
        help="User's input watched size."
    )

    parser.add_argument(
        '--video_embeddings_file',
        type=str,
        default='video_embeddings.vec',
        help='Pretrained video embeddings file.'
    )

    parser.add_argument(
        '--train_file',
        type=str,
        default='train.in',
        help='train data file.'
    )

    parser.add_argument(
        '--validation_file',
        type=str,
        default='validation.in',
        help='validation data file.'
    )

    parser.add_argument(
        '--test_file',
        type=str,
        default='test.in',
        help='test data file.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
