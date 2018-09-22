#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

krank_predict_input_ops = tf.load_op_library('krank_predict_input_ops.so')

sess = tf.Session()
feature_manager_path = '../../fe_dir/feature_manager.bin'

[out1, out2, out3] = krank_predict_input_ops.krank_predict_input(
    watched_rowkeys=[[
        '8575b81e0d8430aj',
        '2975b795930848ah',
        '3845b80a0f8338aj',
        '6595b7bb1da660aj',
        '9495ba0f386641ah',
        '0375b7e5810105aj',
        '3915b8bd706032aj',
        '6235ba13adc772ah',
        '7605b81f1d0614ah',
        '2415b7d59f8191ah',
    ]],
    rinfo1=[[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]],
    rinfo2=[[10.0, 10.0, 1.0, 10.0, 10.0, 1.0, 10.0, 10.0, 1.0, 10.0]],
    target_rowkeys=[['0855ba1f82a336ah', '9575b9ce38d299ah', '7435ba1ad36977aj']],
    is_video=[[True, True, True, True, True, True, True, True, True, True]],
    feature_manager_path=feature_manager_path,
    ws=3
)
out1, out2, out3 = sess.run([out1, out2, out3])

print("positive records: \n", out1)
print("negative records: \n", out2)
print("targets: \n", out3)
