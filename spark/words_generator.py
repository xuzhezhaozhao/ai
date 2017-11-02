
#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: zhezhaoxu
# Created Time : Wed 01 Nov 2017 02:21:19 PM CST
# File Name: words_generator.py
# Description:
"""

import jieba

def load_stop_words():
    words = open("stop_words.csv").read().split('\n')
    uwords = []
    for word in words:
        uwords.append(unicode(word, 'utf-8'))
    return uwords

# tokenize a string
def tokenize(str):
    tokens = [token for token in jieba.cut(str)]
    stop_words = load_stop_words()

    utokens = []
    for token in tokens:
        if token in stop_words:
            continue
        utokens.append(token)
    return utokens

 
def extract(filename):
    df = []

    with open(filename, "rU") as f:
        for line in f:
            s = line.split('\t')
            uin = str(s[1])
            title = s[4]
            content = s[5]
            try:
                lable = int(s[6])
                df.append( (lable, [uin] + tokenize(title + " " + content), ) )
            except:
                print("line: ", cnt)

    return df


if __name__ == "__main__":
    # (label, [words])
    label_words = extract("post_tagged.csv")
    with open("gbar.all", "w") as f:
        for label, words in label_words:
            f.write('__label__' + str(label))
            f.write(' ')
            for word in words:
                f.write(word.encode('utf-8'))
                f.write(' ')
            f.write('\n')
