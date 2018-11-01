import tensorflow as tf
import tensorflow.contrib.slim.nets as nets

slim = tf.contrib.slim
vgg = nets.vgg

inputs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))

with tf.Graph().as_default():
    with slim.arg_scope(vgg.vgg_arg_scope()):
        predictions, endpoints = vgg.vgg_16(
            inputs, is_training=True, num_classes=2)

    print('predictions: ')
    print(predictions)
    print('\nNetword endpoints: ')
    for name in endpoints:
        print('{}: {}'.format(name, endpoints[name]))
