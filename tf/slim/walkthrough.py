
# code from: https://github.com/tensorflow/models/blob/master/research/slim/slim_walkthrough.ipynb


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow as tf
import time

# Main slim library
from tensorflow.contrib import slim


def regression_model(inputs, is_training=True, scope="deep_regression"):
    """Creates the regression model.

    Args:
        inputs: A node that yields a `Tensor` of size [batch_size, dimensions].
        is_training: Whether or not we're currently training the model.
        scope: An optional variable_op scope for the model.

    Returns:
        predictions: 1-D `Tensor` of shape [batch_size] of responses.
        end_points: A dict of end points representing the hidden layers.
    """
    with tf.variable_scope(scope, 'deep_regression', [inputs]):
        end_points = {}
        # Set the default weight _regularizer and acvitation for each
        # fully_connected layer.
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(0.01)):

            # Creates a fully connected layer from the inputs with 32 hidden
            # units.
            net = slim.fully_connected(inputs, 32, scope='fc1')
            end_points['fc1'] = net

            # Adds a dropout layer to prevent over-fitting.
            net = slim.dropout(net, 0.8, is_training=is_training)

            # Adds another fully connected layer with 16 hidden units.
            net = slim.fully_connected(net, 16, scope='fc2')
            end_points['fc2'] = net

            # Creates a fully-connected layer with a single hidden unit.
            # Note that the layer is made linear by setting activation_fn=None.
            predictions = slim.fully_connected(
                net, 1, activation_fn=None, scope='prediction')
            end_points['out'] = predictions

            return predictions, end_points


with tf.Graph().as_default():
    # Dummy placeholders for arbitrary number of 1d inputs and outputs
    inputs = tf.placeholder(tf.float32, shape=(None, 1))
    outputs = tf.placeholder(tf.float32, shape=(None, 1))

    # Build model
    predictions, end_points = regression_model(inputs)

    # Print name and shape of each tensor.
    print("Layers")
    for k, v in end_points.items():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))

    # Print name and shape of parameter nodes  (values not yet initialized)
    print("\n")
    print("Parameters")
    for v in slim.get_model_variables():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))


def produce_batch(batch_size, noise=0.3):
    xs = np.random.random(size=[batch_size, 1]) * 10
    ys = np.sin(xs) + 5 + np.random.normal(size=[batch_size, 1], scale=noise)
    return [xs.astype(np.float32), ys.astype(np.float32)]


x_train, y_train = produce_batch(200)
x_test, y_test = produce_batch(200)


def convert_data_to_tensors(x, y):
    inputs = tf.constant(x)
    inputs.set_shape([None, 1])

    outputs = tf.constant(y)
    outputs.set_shape([None, 1])
    return inputs, outputs


# The following snippet trains the regression model using a mean_squared_error
# loss.
ckpt_dir = './model_dir/'

with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    inputs, targets = convert_data_to_tensors(x_train, y_train)

    # Make the model.
    predictions, nodes = regression_model(inputs, is_training=True)

    # Add the loss function to the graph.
    loss = tf.losses.mean_squared_error(
        labels=targets, predictions=predictions)

    # The total loss is the user's loss plus any regularization losses.
    total_loss = tf.losses.get_total_loss()

    # Specify the optimizer and create the train op:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
    train_op = slim.learning.create_train_op(total_loss, optimizer)
    # Run the training inside a session.
    final_loss = slim.learning.train(
        train_op,
        logdir=ckpt_dir,
        number_of_steps=50000,
        save_summaries_secs=5,
        log_every_n_steps=500)

print("Finished training. Last batch loss:", final_loss)
print("Checkpoint saved in %s" % ckpt_dir)


with tf.Graph().as_default():
    inputs, targets = convert_data_to_tensors(x_train, y_train)
    predictions, end_points = regression_model(inputs, is_training=True)

    # Add multiple loss nodes.
    mean_squared_error_loss = tf.losses.mean_squared_error(
        labels=targets, predictions=predictions)
    absolute_difference_loss = tf.losses.absolute_difference(
        predictions, targets)

    # The following two ways to compute the total loss are equivalent
    regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
    total_loss1 = mean_squared_error_loss \
        + absolute_difference_loss \
        + regularization_loss

    # Regularization Loss is included in the total loss by default.
    # This is good for training, but not for testing.
    total_loss2 = tf.losses.get_total_loss(add_regularization_losses=True)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        total_loss1, total_loss2 = sess.run([total_loss1, total_loss2])

        print('Total Loss1: %f' % total_loss1)
        print('Total Loss2: %f' % total_loss2)

        print('Regularization Losses:')
        for loss in tf.losses.get_regularization_losses():
            print(loss)

        print('Loss Functions:')
        for loss in tf.losses.get_losses():
            print(loss)


with tf.Graph().as_default():
    inputs, targets = convert_data_to_tensors(x_test, y_test)

    # Create the model structure. (Parameters will be loaded below.)
    predictions, end_points = regression_model(inputs, is_training=False)

    global_step = tf.train.get_or_create_global_step()
    # Make a session which restores the old parameters from a checkpoint.
    with tf.train.MonitoredTrainingSession(checkpoint_dir=ckpt_dir) as sess:
        inputs, predictions, targets = sess.run([inputs, predictions, targets])

plt.scatter(inputs, targets, c='r')
plt.scatter(inputs, predictions, c='b')
plt.title('red=true, blue=predicted')
plt.show()


with tf.Graph().as_default():
    inputs, targets = convert_data_to_tensors(x_test, y_test)
    predictions, end_points = regression_model(inputs, is_training=False)

    # Specify metrics to evaluate:
    names_to_value_nodes, names_to_update_nodes = \
        slim.metrics.aggregate_metric_map({
            'Mean Squared Error': slim.metrics.streaming_mean_squared_error(
                predictions, targets),
            'Mean Absolute Error': slim.metrics.streaming_mean_absolute_error(
                predictions, targets)})

    # Make a session which restores the old graph parameters, and then run
    # eval.
    with tf.train.MonitoredTrainingSession(checkpoint_dir=ckpt_dir) as sess:
        metric_values = slim.evaluation.evaluation(
            sess,
            num_evals=1,  # Single pass over data
            eval_op=names_to_update_nodes.values(),
            final_op=names_to_value_nodes.values())

    names_to_values = dict(zip(names_to_value_nodes.keys(), metric_values))
    for key, value in names_to_values.items():
        print('%s: %f' % (key, value))
