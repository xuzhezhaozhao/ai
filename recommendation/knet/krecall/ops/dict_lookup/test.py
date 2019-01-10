#! /usr/bin/env python
# -*- coding=utf8 -*-

import tensorflow as tf

dict_lookup_ops = tf.load_op_library('dict_lookup_ops.so')
dict_lookup = dict_lookup_ops.dict_lookup

test_dict = ['', 'a', 'b', 'c', 'd']
test_data = [['a', 'f', 'f'], ['f', 'c', 'f']]

sess = tf.Session()
ids, num_in_dict = dict_lookup(dict=tf.make_tensor_proto(test_dict),
                               output_ws=2,
                               input=test_data)

ids, num_in_dict = sess.run([ids, num_in_dict])
print(ids)
print(num_in_dict)
