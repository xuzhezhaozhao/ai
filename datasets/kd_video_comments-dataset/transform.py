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

import re
import jieba
import sys

if len(sys.argv) != 3:
    print("Usage: <input> <output>")
    sys.exit(-1)


def getStopWords():
    stop_words = [unicode(word, 'utf-8')
                  for word in open("./dict/stop_words.txt").read().split('\n')]
    return set(stop_words)


if __name__ == "__main__":
    stop_words = getStopWords()

    zh = re.compile(ur'[\u4e00-\u9fa5\u0041-\u005A\u0061-\u007A]+')
    jieba.load_userdict("./dict/jieba_dict.txt")

    with open(sys.argv[2], "w") as fout:
        txt_path = sys.argv[1]
        for line in open(txt_path):
            content = unicode(line.replace('\r\n', ' ').replace(
                '\r', ' ').replace('\n', ' '), 'utf-8').lower()
            content = '#'.join(zh.findall(content))
            stop_words.add('#')
            raw_words = jieba.cut(content, cut_all=False)
            words = [word for word in raw_words if word not in stop_words]
            for word in words:
                fout.write(word.encode('utf-8') + ' ')
            fout.write('\n')
