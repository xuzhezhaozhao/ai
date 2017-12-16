#! /usr/bin/env python
# -*-coding:utf-8 -*-

import struct
import argparse


# Basic model parameters as external flags.
FLAGS = None


def vec2binary(vecfile, targetfile, dictfile):
    with open(targetfile, "wb") as fout:
        with open(dictfile, "w") as fdict:
            for index, line in enumerate(open(vecfile)):
                tokens = line.strip().split(' ')
                if index == 0:
                    nums = map(int, tokens)
                    ff = '<i'
                else:
                    fdict.write(tokens[0] + '\n')
                    nums = map(float, tokens[1:])
                    ff = '<f'
                for num in nums:
                    fout.write(struct.pack(ff, num))


def binary2vec(binaryfile, vecfile, dictfile):
    with open(binaryfile, 'rb') as fbinary:
        with open(vecfile, 'w') as fvec:
            with open(dictfile, 'r') as fdict:
                databytes = fbinary.read()
                total = struct.unpack('<i', databytes[0:4])[0]
                dim = struct.unpack('<i', databytes[4:8])[0]
                fvec.write(str(total))
                fvec.write(' ')
                fvec.write(str(dim))
                fvec.write('\n')
                for x in xrange(total):
                    rowkey = fdict.readline().strip()
                    fvec.write(rowkey + ' ')
                    for y in xrange(dim):
                        offset = 8 + (x * dim + y) * 4
                        num = struct.unpack('<f',
                                            databytes[offset:offset+4])[0]
                        fvec.write(str(num))
                        fvec.write(' ')
                    fvec.write('\n')


def main():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_video',
        type=str,
        default='',
        help='input video tags file.'
    )

    parser.add_argument(
        '--input_article',
        type=str,
        default='',
        help='input article tags file.'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='Output fasttext format.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
