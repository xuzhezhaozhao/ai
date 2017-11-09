#! /usr/bin/env python
#-*- coding=utf8 -*-

import sys

def generate_url(bid, pid):
    return "https://buluo.qq.com/p/detail.html?bid=" + bid + "&pid=" + pid

def parse_title(line):
    tokens = line.split()
    bid_pid = tokens[2].split(":")
    prob = tokens[3]
    bid = bid_pid[0]
    pid = bid_pid[1]
    return (bid, pid, prob)

def parse_line(line):
    tokens = line.split()
    bid_pid = tokens[0].split(":")
    prob = tokens[1]
    bid = bid_pid[0]
    pid = bid_pid[1]
    return (bid, pid, prob)

if __name__ == "__main__":
    input = sys.argv[1]
    output = sys.argv[2]
    with open(output, "w") as fout:
        for line in open(input, "r"):
            if line.find("Query word?") != -1:
                bid, pid, prob = parse_title(line.strip())
                fout.write("\n-----------------------------\n")
            else:
                bid, pid, prob = parse_line(line.strip())

            fout.write( generate_url(bid, pid) )
            fout.write("\n")
            fout.write(prob)
            fout.write("\n")
