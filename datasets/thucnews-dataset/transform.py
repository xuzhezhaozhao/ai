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


basedir = './thucnews'


def getLabels():
    labels = filter(lambda x: os.path.isdir(os.path.join(basedir, x)),
                    os.listdir(basedir))
    return labels


def getStopWords():
    stop_words = [unicode(word, 'utf-8')
                  for word in open(
                          "./thucnews/stop_words.txt").read().split('\n')]
    return set(stop_words)


if __name__ == "__main__":
    labels = getLabels()
    stop_words = getStopWords()

    zh = re.compile(ur'[\u4e00-\u9fa5\u0041-\u005A\u0061-\u007A]+')
    jieba.load_userdict("./thucnews/jieba_dict.txt")

    with open("thucnews.preprocessed", "w") as fthu:
        for label in labels:
            print("preprocess {} ...".format(label))
            txts = filter(lambda x: x.endswith(".txt"),
                          os.listdir(os.path.join(basedir, label)))
            for txt in txts:
                txt_path = os.path.join(basedir, label, txt)
                with open(txt_path) as f:
                    content = unicode(f.read().replace('\r\n', ' ').replace(
                        '\r', ' ').replace('\n', ' '), 'utf-8').lower()
                    content = ' '.join(zh.findall(content))
                    raw_words = jieba.cut(content, cut_all=False)
                    words = [word for word in raw_words
                             if word not in stop_words]
                    fthu.write('__label__' + label + ' ')
                    for word in words:
                        fthu.write(word.encode('utf-8') + ' ')
                    fthu.write('\n')
