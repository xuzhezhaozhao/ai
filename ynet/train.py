
from ynet import YNet
import tensorflow as tf
import argparse
import sys
import os

# Basic model parameters as external flags.
FLAGS = None


def nce_loss(user_vectors,
             predicts,
             embeddings,
             biases,
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
    return tf.reduce_mean(losses)


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
    embedding_dim = FLAGS.embedding_dim
    keep_prob = FLAGS.keep_prob
    learning_rate = FLAGS.learning_rate
    epoches = FLAGS.epoches

    # TODO
    num_videos = 0

    with tf.Graph().as_default():
        video_embeddings = tf.Variable()
        video_biases = tf.Variable()

        # Generate placeholders for input and predicts
        input_placeholder = tf.placeholder(tf.float32,
                                           shape=(batch_size, embedding_dim))
        predicts_placeholder = tf.placeholder(tf.int32,
                                              shape=(batch_size))

        model = YNet(input_placeholder,
                     predicts_placeholder,
                     keep_prob)
        user_vectors = model.user_vectors

        loss = nce_loss(user_vectors,
                        predicts_placeholder,
                        video_embeddings,
                        video_biases,
                        num_videos)
        train_op = training(loss, learning_rate)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in xrange(epoches):
                pass


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

    parser.add_argument(
        '--num_sampled',
        type=int,
        default=12,
        help='number of sampled of NCE loss.'
    )

    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=256,
        help='Video embedding dimension.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
