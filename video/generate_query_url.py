#! /usr/bin/env python
#-*- coding=utf8 -*-

import sys

def generate_url(bid, pid):
    return "https://buluo.qq.com/p/detail.html?bid=" + bid + "&pid=" + pid

def parse_line(line):
    tokens = line.split(":")
    bid = tokens[0]
    pid = tokens[1]
    return (bid, pid)

if __name__ == "__main__":
    input = sys.argv[1]
    output = sys.argv[2]
    with open(output, "w") as fout:
        for line in open(input, "r"):
            bid, pid = parse_line(line.strip())
            fout.write( generate_url(bid, pid) )
            fout.write("\n")
