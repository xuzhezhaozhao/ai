#! /usr/bin/env python
# -*- coding=utf8 -*-


import tensorflow as tf

import vgg19


with tf.Session() as sess:
    images = tf.placeholder("float", [None, 224, 224, 3])
    feed_dict = {images: None}  # TODO

    vgg = vgg19.Vgg19()
    with tf.name_scope("content_vgg"):
        vgg.build(images)

        # TODO feature
        prob = sess.run(vgg.prob, feed_dict=feed_dict)
