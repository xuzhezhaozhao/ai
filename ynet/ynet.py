import tensorflow as tf
import math


class YNet(object):
    """Implementation of Deep Neural Networks for YouTube recommendations."""

    def __init__(self, x, labels, keep_prob):
        """Create the graph of the YNet model
        Args:
            x: Placeholder for the input tensor
            labels: Placeholder for the labels
            keep_prob: Dropout probability
        """
        self._x = x
        self._labels = labels
        self._keep_prob = keep_prob

        # Call the create function to build the computational graph of YNet
        self.create()

    def create(self):
        """Create the network graph."""

        # 1st Layer: FC (w ReLu) -> Dropout
        num_in = self._x.get_shape()[-1].value
        fc1 = fc(self._x, num_in, 2048, name='fc1')
        dropout1 = dropout(fc1, self._keep_prob)

        # 2st Layer: FC (w ReLu) -> Dropout
        fc2 = fc(dropout1, 2048, 1024, name='fc2')
        dropout2 = dropout(fc2, self._keep_prob)

        # 3rd Layer: FC (w ReLu) -> Dropout
        fc3 = fc(dropout2, 1024, 512, name='fc3')
        dropout3 = dropout(fc3, self._keep_prob)

        # 4th Layer: FC (ReLu)
        self._uv = fc(dropout3, 512, 256, name='fc4')

    @property
    def user_vectors(self):
        return self._uv


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer.
    Args:
        x: input tensor
        num_in: number of input neurons
        num_out: number of output neurons
        name:
        relu: Weather apply relu layer

    Returns:
        output:
    """
    with tf.name_scope(name):

        # Create tf variables for the weights and biases
        weights = tf.Variable(
            tf.truncated_normal([num_in, num_out],
                                stddev=1.0 / math.sqrt(float(num_in))),
            name='weights')
        biases = tf.Variable(tf.zeros([num_out]),
                             name='biases')

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=name)
        if relu:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def dropout(x, keep_prob):
    """Create a dropout layer."""

    return tf.nn.dropout(x, keep_prob)
