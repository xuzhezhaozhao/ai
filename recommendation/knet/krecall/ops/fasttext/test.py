#! /usr/bin/env python
# -*- coding=utf8 -*-

import tensorflow as tf

fasttext_example_generate_ops = tf.load_op_library('fasttext_example_generate_ops.so')

test_line = [
    '03959e48258956ah 6635a24a17e289ae 2615a3b2c54096aj 0005a21359a912ae',
    '5585a28c29d816aj 5745a2ce833852ae 9605a3384bf404ag 5535a2e0a04600aj 22559af71bd295ab 9605a3384bf404ag 8605979ada5056ae 5775a2d00fc256ao 6505a28edf8329ai 51959e1901a729ae'
]

sess = tf.Session()

train_data_path = '../../../../data/train_data.in'
(dummy1, dummy2, dummy3) = fasttext_example_generate_ops.fasttext_example_generate(
    train_data_path=train_data_path,
    input=test_line,
    use_saved_dict=False,
    dict_dir="dict_dir",
    ntargets=2
)
sess.run(dummy1)

(records_tensor, labels_tensor, tokens_tensor) = fasttext_example_generate_ops.fasttext_example_generate(
    train_data_path=train_data_path,
    input=test_line,
    use_saved_dict=True,
    dict_dir="dict_dir",
    ntargets=2
)
records, labels, tokens = sess.run([records_tensor, labels_tensor, tokens_tensor])
print("records = \n{}\nlabels = \n{}\ntokens = \n{}".format(records, labels, tokens))

tokens = sess.run(tokens_tensor)
print("records = \n{}\nlabels = \n{}\ntokens = \n{}".format(records, labels, tokens))
