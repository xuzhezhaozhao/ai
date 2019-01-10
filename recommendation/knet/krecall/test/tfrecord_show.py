#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tfrecord_file = 'example.tfrecord'
ws = 20


def parse_example(serialized):
    example = tf.parse_single_example(
        serialized,
        features={
            'records': tf.FixedLenFeature([ws], tf.int64),
            'label': tf.FixedLenFeature([1], tf.int64)
        }
    )
    return (example, example['label'])


ds = tf.data.TFRecordDataset([tfrecord_file])
ds = ds.map(parse_example)

it = ds.make_initializable_iterator()
next_element = it.get_next()
print(next_element)


with tf.Session() as sess:
    sess.run(it.initializer)

    cnt = 10
    while cnt > 0:
        cnt -= 1
        try:
            example = sess.run(next_element)
            print(example)
        except Exception:
            break
