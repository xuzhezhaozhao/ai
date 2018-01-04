#! /usr/bin/env python
# -*-coding:utf-8 -*-

import argparse

# Basic model parameters as external flags.
FLAGS = None


def load_dict(dictfile):
    D = set()
    for line in open(dictfile):
        D.add(line.strip())
    return D


def filter(inputfile, outputfile, D):
    with open(outputfile, 'w') as fout:
        for line in open(inputfile):
            tokens = line.strip().split(' ')
            fout.write(tokens[0])
            for key in tokens[1:]:
                if key in D:
                    fout.write(' ')
                    fout.write(key)
            fout.write('\n')


def main():
    D = load_dict(FLAGS.dictfile)
    filter(FLAGS.input, FLAGS.output, D)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        default='',
        help='Input file.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='Output file.'
    )
    parser.add_argument(
        '--dictfile',
        type=str,
        default='',
        help='Dict file.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
