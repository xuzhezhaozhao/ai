#! /usr/bin/env python
# -*- coding=utf8 -*-

"""
#Author : zhezhaoxu
#Created Time : Wed 01 Nov 2017 03 : 59 : 24 PM CST
#File Name : transform.py
#Description:
"""

# 生成的文本里面有连续空格，用下面的命令压缩空格
# sed 's/[ ][ ]*/ /g'

import os
import re
import jieba
import sys


if len(sys.argv) != 4:
    print("Usage: <input_dir> <output_tokens> <output_raw>")
    sys.exit(-1)
input_dir = sys.argv[1]
output_tokens = sys.argv[2]
output_raw = sys.argv[3]


def getLabels():
    labels = filter(lambda x: os.path.isdir(os.path.join(input_dir, x)),
                    os.listdir(input_dir))
    return labels


def getStopWords():
    stop_words = [unicode(word, 'utf-8')
                  for word in open("./dict/stop_words.txt").read().split('\n')]
    return set(stop_words)


if __name__ == "__main__":
    labels = getLabels()
    stop_words = getStopWords()

    zh = re.compile(ur'[\u4e00-\u9fa5\u0041-\u005A\u0061-\u007A]+')
    jieba.load_userdict("./dict/jieba_dict.txt")

    with open(output_tokens, "w") as ftokens, open(output_raw, 'w') as fraw:
        for label in labels:
            print("preprocess {} ...".format(label))
            txts = filter(lambda x: x.endswith(".txt"),
                          os.listdir(os.path.join(input_dir, label)))
            for txt in txts:
                txt_path = os.path.join(input_dir, label, txt)
                with open(txt_path) as f:
                    raw = unicode(f.read().replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').replace('\t', ' '),
                                  'utf-8')
                    content = ' '.join(zh.findall(raw.lower()))
                    raw_words = jieba.cut(content, cut_all=False)
                    words = [word for word in raw_words
                             if word not in stop_words]
                    ftokens.write('__label__' + label + '\t')
                    for word in words:
                        ftokens.write(word.encode('utf-8') + ' ')
                    ftokens.write('\n')

                    fraw.write('__label__' + label + '\t')
                    fraw.write(raw.encode('utf-8'))
                    fraw.write('\n')
