#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import sys

"""统计 train data 字符数，选出出现次数大于 min-count 的字符."""

if len(sys.argv) != 3:
    print("Usage: <train-data> <min-count>")
    sys.exit(-1)


min_count = int(sys.argv[2])
dic = Counter()
for line in open(sys.argv[1]):
    line = line.strip()
    line = unicode(line, 'utf-8')
    pos = line.find(' \t')
    for ch in line[pos+1:]:
        dic[ch] += 1

for ch in dic:
    if dic[ch] < min_count:
        continue
    print("{}".format(ch.encode('utf-8')))
