#! /usr/bin/env python
# -*- coding=utf8 -*-

import tensorflow as tf


char_cnn_input_ops = tf.load_op_library('./char_cnn_input_ops.so')

sess = tf.Session()

char_ids, label = char_cnn_input_ops.char_cnn_input(
    input=["__label__neg 我 ,。abc"],
    char_dict=tf.make_tensor_proto(['', ' ', ',', '。', '我', 'a', 'b', 'c']),
    label_dict=tf.make_tensor_proto(['__label__pos', '__label__neg']),
    label_str='__label__',
    max_length=8)
char_ids, label = sess.run([char_ids, label])
print("char_ids: ", char_ids)
print("label: ", label)
