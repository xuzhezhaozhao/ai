#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tfrecord_file = 'example.tfrecord'

writer = tf.python_io.TFRecordWriter(tfrecord_file)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


receive_ws = 101
x = [100 for i in range(receive_ws)]
example1 = tf.train.Example(
    features=tf.train.Features(
        feature={
            'words': _int64_feature(x),
        }
    )
).SerializeToString()

for i in range(100):
    writer.write(example1)

writer.close()


def parse_example(serialized):
    example = tf.parse_single_example(
        serialized,
        features={
            'words': tf.FixedLenFeature([receive_ws], tf.int64)
        }
    )
    return (example, example['words'])


ds = tf.data.TFRecordDataset([tfrecord_file])
ds = ds.map(parse_example)

it = ds.make_initializable_iterator()
next_element = it.get_next()
print(next_element)


with tf.Session() as sess:
    sess.run(it.initializer)
    example = sess.run(next_element)

print(example)
