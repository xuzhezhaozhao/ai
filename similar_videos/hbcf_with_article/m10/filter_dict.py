#! /usr/bin/env python
# -*- coding=utf8 -*-


import argparse


# Basic model parameters as external flags.
FLAGS = None


def main():
    fasttext_dict = set()
    video_dict = set()
    subset = set()

    for line in open(FLAGS.input_fasttext_dict_file):
        line = line.strip()
        if line == "":
            continue
        fasttext_dict.add(line)

    for line in open(FLAGS.input_video_dict_file):
        line = line.strip()
        if line == "":
            continue
        video_dict.add(line)

    subset = fasttext_dict.intersection(video_dict)
    with open(FLAGS.output_fasttext_subset_dict_file, "w") as f:
        for s in subset:
            f.write(s)
            f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_fasttext_dict_file',
        type=str,
        default='',
        help=''
    )

    parser.add_argument(
        '--input_video_dict_file',
        type=str,
        default='',
        help=''
    )

    parser.add_argument(
        '--output_fasttext_subset_dict_file',
        type=str,
        default='',
        help=''
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
