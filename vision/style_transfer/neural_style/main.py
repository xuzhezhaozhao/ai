#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
from neural_style import NeuralStyle

import os
import tensorflow as tf
import numpy as np


tf.app.flags.DEFINE_string('model_dir', 'model_dir', '')
tf.app.flags.DEFINE_string('vgg19_npy_path', '', '')

tf.app.flags.DEFINE_string('style_image_path', '', '')
tf.app.flags.DEFINE_string('content_image_path', '', '')
tf.app.flags.DEFINE_string('output_image_path', '', '')
tf.app.flags.DEFINE_bool('use_init_image', False, '')
tf.app.flags.DEFINE_string('init_image_path', '', '')
tf.app.flags.DEFINE_integer('iters', 100, '')

opts = tf.app.flags.FLAGS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'


def main(_):

    neural_style = NeuralStyle(opts)
    loss_tensor = neural_style.loss
    content_loss_tensor = neural_style.content_loss
    style_loss_tensor = neural_style.style_loss
    optimizer = tf.train.AdamOptimizer(
        learning_rate=10.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8)
    train_op = optimizer.minimize(loss=loss_tensor)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in xrange(opts.iters):
            _, loss, content_loss, style_loss = sess.run(
                [train_op, loss_tensor,
                 content_loss_tensor, style_loss_tensor])
            print("iter {}, loss = {:.2f}, content_loss = {:.2f},"
                  " style_loss = {:.2f}"
                  .format(step, loss, content_loss, style_loss))

            if step % 100 == 0:
                # save output image
                img = sess.run([neural_style.output_image])
                img = np.squeeze(img, 0)
                imsave(opts.output_image_path + '.' + str(step) + '.jpg', img)

        # save output image
        img = sess.run([neural_style.output_image])
        img = np.squeeze(img, 0)
        imsave(opts.output_image_path, img)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
