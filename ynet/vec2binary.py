#! /usr/bin/env python
# -*-coding:utf-8 -*-

import struct


def vec2binary(filename, targetfile):
    with open(targetfile, "wb") as fout:
        for index, line in enumerate(open(filename)):
            if index == 0:
                # TODO write total numbers
                continue
            tokens = line.strip().split(' ')
            nums = map(float, tokens[1:])
            for num in nums:
                print num
                bytes = struct.pack('<f', num)
                fout.write(bytes)


def check(filename):
    with open(filename) as f:
        databytes = f.read()
        num = struct.unpack('<f', databytes[0:4])
        print num


def main():
    vec2binary('mini.vec', 'mini.binary')


if __name__ == "__main__":
    main()
