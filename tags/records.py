#! /usr/bin/env python
# -*- coding=utf8 -*-

import argparse
from tags import delteduplicated
from tags import load_tag_info_dict

# Basic model parameters as external flags.
FLAGS = None


def load_rowkey2tagids_info(inputfile, rowkey2tagids):
    """
    rowkey2tagids: python dict
    """
    for index, line in enumerate(open(inputfile, 'r')):
        if index == 0:
            continue
        tokens = line.strip().split('/')
        rowkey = tokens[0]
        if rowkey in rowkey2tagids:
            print("[W] duplicated rowkey")
            continue
        tagids = tokens[1].split(',') + tokens[2].split(',')
        tagids = filter(lambda x: x != '', tagids)
        tagids = map(int, tagids)
        tagids = delteduplicated(tagids)
        if FLAGS.sort_tags:
            tagids.sort()
        rowkey2tagids[rowkey] = tagids


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
            for rowkey in history:
                fout.write(rowkey + " ")
            fout.write("\n")

            if index % 500000 == 0:
                print(str(index) + " lines writen ...")


def write_histories_tagsid(histories):
    rowkey2tagids = dict()
    load_rowkey2tagids_info(FLAGS.input_article_tags_file, rowkey2tagids)
    load_rowkey2tagids_info(FLAGS.input_video_tags_file, rowkey2tagids)
    taginfo = load_tag_info_dict(FLAGS.input_tag_info_file)
    with open(FLAGS.output_history_tags, "w") as fout:
        for index, uin in enumerate(histories):
            history = histories[uin]
            sz = len(history)
            if sz < FLAGS.min_items or sz > FLAGS.max_items:
                continue
            fout.write("__label__" + uin + " ")
            for rowkey in history:
                if rowkey not in rowkey2tagids:
                    print("[W] rowken not in rowkey2tagids")
                    continue
                tagids = rowkey2tagids[rowkey]
                for tagid in tagids:
                    if tagid in taginfo:
                        fout.write(taginfo[tagid] + " ")
                    else:
                        fout.write(str(tagid) + " ")
            fout.write("\n")

            if index % 500000 == 0:
                print(str(index) + " lines writen ...")


def main():
    histories = convert2histories()

    if FLAGS.output_history_raw != "":
        write_histories_raw(histories)

    write_histories_tagsid(histories)


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

    parser.add_argument(
        '--sort_tags',
        type=bool,
        default=True,
        help=''
    )

    parser.add_argument(
        '--input_tag_info_file',
        type=str,
        default='',
        help='input tag info file.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
