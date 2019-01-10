#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys


if len(sys.argv) != 3:
    print("Usage: <predict_1> <predict_2>")
    sys.exit(-1)


for p1, p2 in zip(open(sys.argv[1]), open(sys.argv[2])):
    label1, s1 = p1.strip().split()
    label2, s2 = p2.strip().split()
    s1, s2 = float(s1), float(s2)

    if label1 == label2:
        s = (s1 + s2) / 2.0
        label = label1
    else:
        s = (s1 + (1 - s2)) / 2.0
        if s > 0.5:
            label = label1
        else:
            s = 1 - s
            label = label2
    print("{} {}".format(label, s))
