#! /usr/bin/env python
# -*- coding=utf8 -*-

"""
#Author : zhezhaoxu
#Created Time : Wed 01 Nov 2017 03 : 59 : 24 PM CST
#File Name : transform.py
#Description:
"""

import os
import sys


if len(sys.argv) != 4:
    print("Usage: <input_dir> <output> <label>")
    sys.exit(-1)
input_dir = sys.argv[1]
output_raw = sys.argv[2]
label_file = sys.argv[3]


def getLabels():
    labels = filter(lambda x: os.path.isdir(os.path.join(input_dir, x)),
                    os.listdir(input_dir))
    return labels


if __name__ == "__main__":
    labels = getLabels()
    with open(output_raw, 'w') as fraw, open(label_file, 'w') as flabel:
        for label in labels:
            flabel.write('__label__' + label + '\n')
            print("preprocess {} ...".format(label))
            txts = filter(lambda x: x.endswith(".txt"),
                          os.listdir(os.path.join(input_dir, label)))
            for txt in txts:
                txt_path = os.path.join(input_dir, label, txt)
                with open(txt_path) as f:
                    raw = unicode(f.read().replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').replace('\t', ' '),
                                  'utf-8')
                    fraw.write('__label__' + label + '\t')
                    fraw.write(raw.encode('utf-8'))
                    fraw.write('\n')
