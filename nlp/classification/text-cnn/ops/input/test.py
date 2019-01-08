#! /usr/bin/env python
# -*- coding=utf8 -*-

import tensorflow as tf


text_cnn_input_ops = tf.load_op_library('./text_cnn_input_ops.so')

sess = tf.Session()

word_ids, label = text_cnn_input_ops.text_cnn_input(
    input=["__label_pos a b c d"],
    word_dict=tf.make_tensor_proto(['', 'a', 'b', 'c', 'd']),
    label_dict=tf.make_tensor_proto(['__label_pos', '__label_neg']),
    max_length=8
)
word_ids, label = sess.run(word_ids), sess.run(label)
print("word_ids: ", word_ids)
print("label: ", label)
