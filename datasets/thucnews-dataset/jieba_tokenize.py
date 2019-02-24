#! /usr/bin/env python
# -*- coding=utf8 -*-

# 生成的文本里面有连续空格，用下面的命令压缩空格
# sed 's/[ ][ ]*/ /g'

import re
import jieba
import sys


if len(sys.argv) != 3:
    print("Usage: <input> <output>")
    sys.exit(-1)
input_file = sys.argv[1]
output_file = sys.argv[2]


def getStopWords():
    stop_words = [unicode(word, 'utf-8')
                  for word in open("./dict/stop_words.txt").read().split('\n')]
    return set(stop_words)


if __name__ == "__main__":
    stop_words = getStopWords()

    zh = re.compile(ur'[\u4e00-\u9fa5\u0041-\u005A\u0061-\u007A]+')
    jieba.load_userdict("./dict/jieba_dict.txt")

    with open(output_file, "w") as fout:
        for idx, line in enumerate(open(input_file)):
            if idx % 10000 == 0:
                print("Processed {} lines ...".format(idx))
            line = unicode(line.strip(), 'utf-8')
            tokens = line.split('\t')
            assert(len(tokens) == 2)
            label = tokens[0]
            text = tokens[1]
            text = ' '.join(zh.findall(text.lower()))
            raw_words = jieba.cut(text, cut_all=False)
            words = [word for word in raw_words
                     if word not in stop_words]
            fout.write('__label__' + label + '\t')
            for word in words:
                fout.write(word.encode('utf-8') + ' ')
            fout.write('\n')
