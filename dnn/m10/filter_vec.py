#! /usr/bin/env python
# -*- coding=utf8 -*-


import argparse


FLAGS = None


def main():
    subset = set()

    for line in open(FLAGS.input_fasttext_subset_dict_file):
        line = line.strip()
        if line == "":
            continue
        subset.add(line)

    with open(FLAGS.output_fasttext_subset_vec_file, 'w') as f:
        dim = 0
        for lineindex, line in enumerate(open(FLAGS.input_fasttext_vec_file)):
            line = line.strip()
            tokens = line.split()
            if lineindex == 0:
                dim = int(tokens[1])
                f.write(str(len(subset)))
                f.write(' ')
                f.write(str(dim))
                f.write('\n')
                continue

            if line == "":
                continue

            if tokens[0] in subset:
                f.write(line)
                f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_fasttext_vec_file',
        type=str,
        default='',
        help=''
    )

    parser.add_argument(
        '--input_fasttext_subset_dict_file',
        type=str,
        default='',
        help=''
    )

    parser.add_argument(
        '--output_fasttext_subset_vec_file',
        type=str,
        default='',
        help=''
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
