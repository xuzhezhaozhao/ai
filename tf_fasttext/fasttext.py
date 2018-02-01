
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import os
import sys
import threading
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

fasttext_model = tf.load_op_library(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'fasttext_ops.so'))

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model.")
flags.DEFINE_string("train_data", None, "train data file")
flags.DEFINE_integer("embedding_size", 100, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 5,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 25,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 50,
                     "Numbers of training examples each step processes "
                     "(no minibatching).")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 5,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-4,
                   "Subsample threshold for word occurrence. Words that "
                   "appear with higher frequency will be randomly "
                   "down-sampled. Set to 0 to disable.")

FLAGS = flags.FLAGS


class Options(object):
    """Options used by our word2vec model"""

    def __init__(self):
        # Model options

        # Embedding dimension
        self.emb_dim = FLAGS.embedding_size

        # Training options

        # The training text file
        self.train_data = FLAGS.train_data

        # Number of negative samples per example.
        self.num_samples = FLAGS.num_neg_samples

        # The initial learning rate.
        self.learning_rate = FLAGS.learning_rate

        # Number of epochs to train. After these many epochs, the learning
        # rate decays linearly to zero and the training stops.
        self.epochs_to_train = FLAGS.epochs_to_train

        # Concurrent training steps.
        self.concurrent_steps = FLAGS.concurrent_steps

        # Number of examples for one training step.
        self.batch_size = FLAGS.batch_size

        # The number of words to predict to the left and right of the target
        # word.
        self.window_size = FLAGS.window_size

        # The minimum number of word occurrences for it to be included in the
        # vocabulary.
        self.min_count = FLAGS.min_count

        # Subsampling threshold for word occurrence.
        self.subsample = FLAGS.subsample

        # Where to write out summaries.
        self.save_path = FLAGS.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


class Word2Vec(object):
    """Word2Vec model (xcbow)"""

    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._word2id = {}
        self._id2word = []
        self.build_graph()

    def build_graph(self):
        """Build the model graph."""
        opts = self._options

        # The training data. A text file.
        (words, counts, words_per_epoch, current_epoch,
         total_words_processed, examples, labels,
         valid_lengths) = fasttext_model.fasttext(train_data=opts.train_data,
                                                  batch_size=opts.batch_size,
                                                  seed=11)

        (opts.vocab_words, opts.vocab_counts,
         opts.words_per_epoch) = self._session.run([words, counts,
                                                    words_per_epoch])

        opts.vocab_size = len(opts.vocab_words)
        print("Data file: ", opts.train_data)
        print("Vocab size: ", opts.vocab_size)
        print("Words per epoch: ", opts.words_per_epoch)

        self._id2word = opts.vocab_words
        for i, w in enumerate(self._id2word):
            self._word2id[w] = i

        # Declare all variables we need.
        # Input words embedding: [vocab_size, emb_dim]
        w_in = tf.Variable(
            tf.random_uniform([opts.vocab_size, opts.emb_dim],
                              -0.5 / opts.emb_dim, 0.5 / opts.emb_dim),
            name="w_in")

        # TODO valid_lengths
        embed = tf.nn.embedding_lookup(w_in, examples)
        mean_inputs = tf.reduce_mean(embed, 1)

        # Output
        nce_weights = tf.Variable(
            tf.zeros([opts.vocab_size, opts.emb_dim]), name="nce_weights")

        nce_weights = tf.Variable(
            tf.truncated_normal([opts.vocab_size, opts.emb_dim],
                                stddev=1.0 / math.sqrt(opts.emb_dim)))
        nce_biases = tf.Variable(tf.zeros([opts.vocab_size]),
                                 name="nce_biases")

        # Global step: scalar, i.e., shape [].
        global_step = tf.Variable(0, name="global_step")

        # Linear learning rate decay.
        words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
        lr = opts.learning_rate * tf.maximum(
            0.0001,
            1.0 - tf.cast(total_words_processed, tf.float32) / words_to_train)

        # NCE loss
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=labels,
                inputs=mean_inputs,
                num_sampled=opts.num_samples,
                num_classes=opts.vocab_size
            ),
            name='loss'
        )
        optimizer = tf.train.GradientDescentOptimizer(lr)

        # Training nodes.
        inc = global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            train = optimizer.minimize(loss)

        self._w_in = w_in
        self._examples = examples
        self._labels = labels
        self._lr = lr
        self._train = train
        self._global_step = global_step
        self._epoch = current_epoch
        self._words = total_words_processed
        self._loss = loss

        # Properly initialize all variables.
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()

    def train(self):
        """Train the model."""
        initial_epoch, initial_words = self._session.run([self._epoch,
                                                          self._words])
        while True:
            _, epoch, global_step, loss = self._session.run([self._train,
                                                             self._epoch,
                                                             self._global_step,
                                                             self._loss])
            if epoch != initial_epoch:
                break
            if global_step % 100 == 0:
                print("loss = {}".format(loss))


def main(_):
    """Train a fasttext model."""
    if not FLAGS.train_data or not FLAGS.save_path:
        print("--train_data and --save_path must be specified.")
        sys.exit(1)
    opts = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        model = Word2Vec(opts, session)
        for _ in xrange(opts.epochs_to_train):
            model.train()  # Process one epoch
        # Perform a final save.
        model.saver.save(session, os.path.join(opts.save_path, "model.ckpt"),
                         global_step=model.global_step)


if __name__ == "__main__":
    tf.app.run()
