import tensorflow as tf
from nets import inception

slim = tf.contrib.slim

inputs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))

with tf.Graph().as_default():
    with slim.arg_scope(inception.inception_v4_arg_scope()):
        predictions, endpoints = inception.inception_v4(
            inputs, is_training=True, num_classes=2)

    tf.logging.info('predictions: ')
    tf.logging.info(predictions)
    tf.logging.info('\nNetword endpoints: ')
    for name in endpoints:
        print('{}: {}'.format(name, endpoints[name]))
