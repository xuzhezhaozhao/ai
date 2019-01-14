#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import os
import scipy.misc
import tensorflow as tf
import numpy as np

from neural_style import NeuralStyle


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
    style_image = imread(opts.style_image_path)
    content_image = imread(opts.content_image_path)
    init_image = content_image
    if opts.use_init_image:
        init_image = imread(opts.init_image_path)
        init_image = scipy.misc.imresize(init_image, content_image.shape[:2])

    neural_style = NeuralStyle(opts.vgg19_npy_path,
                               content_image,
                               style_image,
                               init_image)
    loss_tensor = neural_style.loss
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0)
    train_op = optimizer.minimize(loss=loss_tensor)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in xrange(opts.iters):
            _, loss = sess.run([train_op, loss_tensor ])
            print("iter {}, loss {}".format(i, loss))

        # save output image
        img = sess.run([neural_style.output_image])
        img = np.squeeze(img, 0)
        imsave(opts.output_image_path, img)


def imread(img_path):
    img = scipy.misc.imread(img_path).astype(np.float32)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:, :, :3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
