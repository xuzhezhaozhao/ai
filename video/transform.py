#! /usr/bin/env python
#-*- coding=utf8 -*-

import sys

if __name__ == "__main__":
    histories = dict()
    input = sys.argv[1]
    output = sys.argv[2]
    for index, line in enumerate(open(input, "r")):
        tokens = line.strip().split(',')
        uin = tokens[0]
        pid = tokens[1]
        bid = tokens[2]
        time = tokens[3]
        if uin == "" or pid == "" or bid == "" or time == "":
            continue
        if uin not in histories:
            histories[uin] = []
        histories[uin].append(bid + ":" + pid)
        if index % 200000 == 0:
            print(str(index) + " lines processed ...")

    print("write to file ...")
    with open(output, "w") as fout:
        for uin in histories:
            history = histories[uin]
            fout.write("__label__" + uin)
            fout.write(" ")
            for item in history:
                fout.write(item)
                fout.write(" ")
            fout.write("\n")
