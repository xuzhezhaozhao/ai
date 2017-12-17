#! /usr/bin/env python
# -*-coding:utf-8 -*-

import argparse


# Basic model parameters as external flags.
FLAGS = None


def load_tag_info_dict():
    taginfo = {}
    for index, line in enumerate(open(FLAGS.input_tag_info_file, 'r')):
        if index == 0:
            # skip header
            continue
        tokens = line.strip().split('/')
        try:
            taginfo[int(tokens[0])] = tokens[1]
        except Exception:
            pass

    return taginfo


def convertfile(fout, inputfile, taginfo):
    nwarning = 0
    lack_labels = set()
    for index, line in enumerate(open(inputfile, 'r')):
        if index == 0:
            continue
        tokens = line.strip().replace('/', ',').split(',')
        tokens = filter(lambda x: x != '', tokens)
        tokens = map(int, tokens)
        tokens = list(set(tokens))
        tokens.sort()
        if len(tokens) < FLAGS.min_labels:
            continue
        for tag in tokens:
            if tag not in taginfo:
                nwarning += 1
                lack_labels.add(tag)
                continue
            fout.write(taginfo[tag])
            fout.write(' ')
        fout.write('\n')

    print("lack info of labels [{}]: {}".format(nwarning, lack_labels))


def convert():
    taginfo = load_tag_info_dict()
    with open(FLAGS.output, 'w') as fout:
        if FLAGS.input_video_tags_file != '':
            convertfile(fout, FLAGS.input_video_tags_file, taginfo)
        if FLAGS.input_article_tags_file != '':
            convertfile(fout, FLAGS.input_article_tags_file, taginfo)


def main():
    convert()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_video_tags_file',
        type=str,
        default='',
        help='input video tags file.'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_article_tags_file',
        type=str,
        default='',
        help='input article tags file.'
    )

    parser.add_argument(
        '--input_tag_info_file',
        type=str,
        default='',
        help='input tag info file.'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='Output fasttext format.'
    )

    parser.add_argument(
        '--min_labels',
        type=int,
        default=1,
        help=''
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
