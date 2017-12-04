
from ynet import YNet
import tensorflow as tf
import argparse
import sys
import os

# Basic model parameters as external flags.
FLAGS = None


def run_training():
    """Train YNet for a number of epoches."""

    with tf.Graph().as_default():
        # Generate placeholders for the embeddings and labels.
        embeddings_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size, FLAGS.input_size)

        model = YNet(embeddings_placeholder,
                     labels_placeholder,
                     FLAGS.keep_prob)
        train_op = model.trainning(FLAGS.learning_rate)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in xrange(FLAGS.epoch):
                pass


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    run_training()


def placeholder_inputs(batch_size, size):
    """Generate placeholder variables to represent the input tensors

    Args:
        batch_size: The batch size will be baked into both placeholders.
        size: Embeddings size.

    Return:
        embeddings_placeholder: Embeddings placeholder.
        labels_placeholder: Labels placeholder.
    """
    embeddings_placeholder = tf.placeholder(tf.float32,
                                            shape=(batch_size, size))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return embeddings_placeholder, labels_placeholder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )

    parser.add_argument(
        '--epoch',
        type=int,
        default=20,
        help='Number of epoch to run trainer.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
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

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
