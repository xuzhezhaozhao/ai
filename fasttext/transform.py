#! /usr/bin/env python
#-*- coding=utf8 -*-
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

def getLabels():
    labels = filter(lambda x : os.path.isdir(x), os.listdir('.'))
    return labels

def getStopWords():
    stop_words = [unicode(word, 'utf-8') for word in open("stop_words.txt").read().split('\n')]
    return stop_words
    

if __name__ == "__main__":
    labels = getLabels()
    stop_words = getStopWords()

    zh =re.compile(ur'[\u4e00-\u9fa5\u0041-\u005A\u0061-\u007A]+')
    jieba.load_userdict("THUOCL_all.txt")

    with open("thucnew.all", "w") as fthu:
        for label in labels:
            print label
            txts = filter(lambda x : x.endswith(".txt"), os.listdir(label))
            for txt in txts:
                txt_path = os.path.join(label, txt)
                with open(txt_path) as f:
                    content = unicode(f.read().replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' '), 'utf-8').lower()
                    content = ' '.join(zh.findall(content))
                    raw_words = jieba.lcut(content, cut_all=False)
                    words = [word for word in raw_words if word not in stop_words]
                    fthu.write('__label__' + label)
                    fthu.write(' ')
                    for word in words:
                        fthu.write(word.encode('utf-8'))
                        fthu.write(' ')
                    fthu.write('\n')
