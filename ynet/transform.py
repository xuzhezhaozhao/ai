#! /usr/bin/env python
# -*- coding=utf8 -*-

import sys
import os


histories = dict()
input = sys.argv[1]
output = sys.argv[2]
kmin = int(sys.argv[3])
maxlines = -1

if len(sys.argv) == 5:
    maxlines = int(sys.argv[4])

for index, line in enumerate(open(input, "r")):
    if maxlines != -1 and index >= maxlines:
        break

    try:
        tokens = line.strip().split(',')
        time = tokens[0]
        uin = tokens[1]
        rowkey = tokens[2]
        isvideo = int(tokens[3])

        # TODO age 和 gender 很多是 "", redesign try ... except ...
        # age = int(tokens[4])
        # gender = int(tokens[5])

        if time == "" or uin == "" or rowkey == "" or isvideo == "":
            continue

        if uin not in histories:
            histories[uin] = []
        histories[uin].append(rowkey)

        if index % 2000000 == 0:
            print(str(index) + " lines processed ...")
    except Exception as e:
        print("{}: {}".format(index, e))

print("write to file ...")
print("histories size: ", len(histories))
with open(output, "w") as fout:
    for index, uin in enumerate(histories):
        history = histories[uin]
        if len(history) < kmin:
            continue
        fout.write("__label__" + uin + " ")
        for item in history:
            fout.write(item + " ")
        fout.write("\n")
        if index % 500000 == 0:
            print(str(index) + " lines writen ...")
print("write successfully.")

os._exit(0)
