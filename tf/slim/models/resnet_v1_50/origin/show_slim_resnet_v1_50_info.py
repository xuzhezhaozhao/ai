import tensorflow as tf
from nets import resnet_v1

slim = tf.contrib.slim

inputs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))

with tf.Graph().as_default():
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        predictions, endpoints = resnet_v1.resnet_v1_50(
            inputs, is_training=True, num_classes=2)

    tf.logging.info('predictions: ')
    tf.logging.info(predictions)
    tf.logging.info('\nNetword endpoints: ')
    for name in endpoints:
        print('{}: {}'.format(name, endpoints[name]))
