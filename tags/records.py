#! /usr/bin/env python
# -*- coding=utf8 -*-

import argparse

# Basic model parameters as external flags.
FLAGS = None


def load_rowkey2tagids_info(inputfile):
    rowkey2tagids = dict()
    for index, line in enumerate(open(inputfile, 'r')):
        pass

    return rowkey2tagids


def convert2histories():
    histories = dict()
    for index, line in enumerate(open(FLAGS.input, "r")):
        if FLAGS.max_lines != -1 and index >= FLAGS.max_lines:
            break
        tokens = line.strip().split(',')
        time = tokens[0]
        uin = tokens[1]
        rowkey = tokens[2]

        if time == "" or uin == "" or rowkey == "":
            continue

        if uin not in histories:
            histories[uin] = []
        if rowkey not in histories[uin]:
            histories[uin].append(rowkey)

        if index % 2000000 == 0:
            print(str(index) + " lines processed ...")

    return histories


def write_histories_raw(histories):
    with open(FLAGS.output_history_raw, "w") as fout:
        for index, uin in enumerate(histories):
            history = histories[uin]
            sz = len(history)
            if sz < FLAGS.min_items or sz > FLAGS.max_items:
                continue
            fout.write("__label__" + uin + " ")
            for item in history:
                fout.write(item + " ")
            fout.write("\n")

            if index % 500000 == 0:
                print(str(index) + " lines writen ...")


def write_histories_tagsid(histories):
    pass


def main():
    histories = convert2histories()
    if FLAGS.output_history_raw != "":
        write_histories_raw(histories)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        default='',
        help='input records file.'
    )

    parser.add_argument(
        '--input_article_tags_file',
        type=str,
        default='',
        help='input article tags file.'
    )

    parser.add_argument(
        '--input_video_tags_file',
        type=str,
        default='',
        help='input video tags file.'
    )

    parser.add_argument(
        '--max_lines',
        type=int,
        default=-1,
        help='max lines should be processed, -1 means unlimited.'
    )

    parser.add_argument(
        '--output_history_raw',
        type=str,
        default="",
        help='history in rowkey format.'
    )

    parser.add_argument(
        '--min_items',
        type=int,
        default=1,
        help='min items.'
    )

    parser.add_argument(
        '--max_items',
        type=int,
        default=1024,
        help='max items.'
    )

    parser.add_argument(
        '--output_history_tags',
        type=str,
        default="",
        help='history in tags format.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
