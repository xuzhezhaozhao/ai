#! /usr/bin/env python
#-*- coding=utf8 -*-

import sys

if __name__ == "__main__":
    histories = dict()
    input = sys.argv[1]
    output = sys.argv[2]
    query = sys.argv[3]
    klimit = int(sys.argv[4])
    maxlines = -1
    if len(sys.argv) == 6:
        maxlines = int(sys.argv[5])

    key2int = dict()
    keys = list()
    for index, line in enumerate(open(input, "r")):
        if maxlines != -1 and index >= maxlines:
            break
        try:
            tokens = line.strip().split(',')
            uin = tokens[0]
            pid = tokens[1]
            bid = tokens[2]
            time = tokens[3]
            duration = int(tokens[4])
            play_time = int(tokens[5])
            play_ratio = play_time * 1.0 / duration

            if uin == "" or pid == "" or bid == "" or time == "":
                continue
            if play_time < 20 and play_ratio < 0.3:
                continue

            key = bid + ":" + pid
            if key not in key2int:
                key2int[key] = len(key2int)
                keys.append(key)
            pos = key2int[key]
            if uin not in histories:
                histories[uin] = []
            cnt = len(histories[uin])
            if cnt >= klimit:
                histories[uin].pop(0)
            histories[uin].append(pos)
            if index % 2000000 == 0:
                print(str(index) + " lines processed ...")
        except:
            pass

    print("write keys to file ...")
    with open(query, "w") as fout:
        for key in keys:
            fout.write(key)
            fout.write("\n")

    print("write histories to file ...")
    with open(output, "w") as fout:
        for uin in histories:
            history = histories[uin]
            fout.write("__label__" + uin + " ")
            for item in history:
                fout.write(str(item) + " ")
            fout.write("\n")
