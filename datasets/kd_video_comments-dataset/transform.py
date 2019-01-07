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

    # 中文, 英文, 数字
    # https://zh.wikipedia.org/wiki/Unicode%E5%AD%97%E7%AC%A6%E5%88%97%E8%A1%A8
    zh = re.compile(
        ur'[\u4e00-\u9fa5\u0041-\u005A\u0061-\u007A\u0030-\u0039]+')
    jieba.load_userdict("./dict/jieba_dict.txt")

    with open(sys.argv[2], "w") as fout:
        txt_path = sys.argv[1]
        for index, line in enumerate(open(txt_path)):
            if (index+1) % 1000000 == 0:
                print("{} lines tokenized ...".format(index+1))
            content = unicode(line.replace('\r\n', ' ').replace(
                '\r', ' ').replace('\n', ' '), 'utf-8').lower()
            content = '#'.join(zh.findall(content))
            stop_words.add('#')
            raw_words = jieba.cut(content, cut_all=False)
            words = [word for word in raw_words if word not in stop_words]
            for word in words:
                fout.write(word.encode('utf-8') + ' ')
            fout.write('\n')
