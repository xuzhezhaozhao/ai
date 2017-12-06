#! /usr/bin/env python
#-*- coding=utf8 -*-

import sys
import os

if __name__ == "__main__":
    histories = dict()
    input = sys.argv[1]
    output = sys.argv[2]
    klimit = int(sys.argv[3])
    maxlines = -1
    if len(sys.argv) == 5:
        maxlines = int(sys.argv[4])

    nfilter = 0
    for index, line in enumerate(open(input, "r")):
        if maxlines != -1 and index >= maxlines:
            break
        try:
            tokens = line.strip().split(',')
            uin = tokens[0]
            bid = tokens[1]
            pid = tokens[2]
            time = tokens[3]
            duration = int(tokens[4])
            play_time = int(tokens[5])
            play_ratio = play_time * 1.0 / duration
            if uin == "" or pid == "" or bid == "" or time == "":
                continue
            if play_ratio < 0.2:
                ++nfilter
            if uin not in histories:
                histories[uin] = []
            cnt = len(histories[uin])
            if cnt >= klimit:
                histories[uin].pop(0)
            histories[uin].append(bid + ":" + pid)
            if index % 2000000 == 0:
                print(str(index) + " lines processed ...")
        except:
            pass

    print("nfilter: ", nfilter)
    print("write to file ...")
    print("histories size: ", len(histories))
    with open(output, "w") as fout:
        for index, uin in enumerate(histories):
            history = histories[uin]
            fout.write("__label__" + uin + " ")
            for item in history:
                fout.write(item + " ")
            fout.write("\n")
            if index % 500000 == 0:
                print(str(index) + " lines writen ...")
    print("end")
    os._exit(0)
