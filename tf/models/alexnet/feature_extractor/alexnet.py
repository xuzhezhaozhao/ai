import os
import tensorflow as tf
import numpy as np
import time
import inspect


class AlexNet:
    def __init__(self, alexnet_npy_path=None):
        if alexnet_npy_path is None:
            path = inspect.getfile(AlexNet)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "bvlc_alexnet.npy")
            alexnet_npy_path = path
            print(alexnet_npy_path)

        self.data_dict = np.load(alexnet_npy_path, encoding='bytes').item()
        print("npy file loaded")

    def build(self, bgr):
        """
        load variable from npy to build the AlexNet

        :param bgr: bgr image [batch, height, width, 3]
        """

        start_time = time.time()
        print("build model started")
        assert bgr.get_shape().as_list()[1:] == [227, 227, 3]

        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        self.conv1 = self.conv_layer(bgr, 11, 11, 96, 4, 4,
                                     padding='VALID', name='conv1')
        self.relu1 = tf.nn.relu(self.conv1)
        self.norm1 = self.lrn(self.relu1, 2, 2e-05, 0.75, name='norm1')
        self.pool1 = self.max_pool(self.norm1, 3, 3, 2, 2, padding='VALID',
                                   name='pool1')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        self.conv2 = self.conv_layer(self.pool1, 5, 5, 256, 1, 1,
                                     groups=2, name='conv2')
        self.relu2 = tf.nn.relu(self.conv2)
        self.norm2 = self.lrn(self.relu2, 2, 2e-05, 0.75, name='norm2')
        self.pool2 = self.max_pool(self.norm2, 3, 3, 2, 2, padding='VALID',
                                   name='pool2')

        # 3rd Layer: Conv (w ReLu)
        self.conv3 = self.conv_layer(self.pool2, 3, 3, 384, 1, 1, name='conv3')
        self.relu3 = tf.nn.relu(self.conv3)

        # 4th Layer: Conv (w ReLu) splitted into two groups
        self.conv4 = self.conv_layer(self.relu3, 3, 3, 384, 1, 1, groups=2,
                                     name='conv4')
        self.relu4 = tf.nn.relu(self.conv4)

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        self.conv5 = self.conv_layer(self.relu4, 3, 3, 256, 1, 1, groups=2,
                                     name='conv5')
        self.relu5 = tf.nn.relu(self.conv5)
        self.pool5 = self.max_pool(self.relu5, 3, 3, 2, 2, padding='VALID',
                                   name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(self.pool5, [-1, 6*6*256])
        self.fc6 = self.fc_layer(flattened, name='fc6')
        self.relu6 = tf.nn.relu(self.fc6)
        self.fc7 = self.fc_layer(self.relu6, name='fc7')
        self.relu7 = tf.nn.relu(self.fc7)
        self.fc8 = self.fc_layer(self.relu7, name='fc8')
        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def conv_layer(self, x, filter_height, filter_width, num_filters,
                   stride_y, stride_x, name, padding='SAME', groups=1):
        """Create a convolution layer.

        Adapted from: https://github.com/ethereon/caffe-tensorflow
        """

        # Create lambda function for the convolution
        def convolve(i, k): return tf.nn.conv2d(
                i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

        with tf.variable_scope(name):
            # Create tf variables for the weights and biases of the conv layer
            weights = self.get_conv_filter(name)
            biases = self.get_bias(name)

        if groups == 1:
            conv = convolve(x, weights)

        # In the cases of multiple groups, split inputs & weights and
        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                     value=weights)
            output_groups = [convolve(i, k)
                             for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(axis=3, values=output_groups)

        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return bias

    def fc_layer(self, x, name):
        """Create a fully connected layer."""

        with tf.variable_scope(name):
            weights = self.get_conv_filter(name)
            biases = self.get_bias(name)

            # Matrix multiply weights and inputs and add bias
            act = tf.nn.xw_plus_b(x, weights, biases)
            return act

    def max_pool(self, x, filter_height, filter_width,
                 stride_y, stride_x, name, padding='SAME'):
        """Create a max pooling layer."""

        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)

    def lrn(self, x, radius, alpha, beta, name, bias=1.0):
        """Create a local response normalization layer."""

        return tf.nn.local_response_normalization(
            x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias,
            name=name)

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")
