
from ynet import YNet
import tensorflow as tf

batch_size = 50
size = 256
keep_prob = 0.5


def main():
    input_placeholder = placeholder_inputs(batch_size, size)
    net = YNet(input_placeholder, keep_prob)
    graph_output = net.create()


def placeholder_inputs(batch_size, size):
    """Generate placeholder variables to represent the input tensor

    Args:
        batch_size:
        size:

    Return:
    """
    placeholder = tf.placeholder(tf.float32, shape=(batch_size, size))
    return placeholder


if __name__ == '__main__':
    main()
