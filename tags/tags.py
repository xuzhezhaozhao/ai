#! /usr/bin/env python
# -*-coding:utf-8 -*-

import argparse


# Basic model parameters as external flags.
FLAGS = None


def load_tag_info_dict():
    taginfo = {}
    for index, line in enumerate(open(FLAGS.input_tag_info_file, 'r')):
        line = unicode(line, 'utf-8')
        if index == 0:
            # skip header
            continue
        tokens = line.strip().split('/')
        try:
            tagid = int(tokens[0])
            tagname = reduce(lambda x, y: x+y, tokens[1:-1])
            taginfo[tagid] = tagname.replace(' ', '_')
        except Exception:
            pass

    return taginfo


def convertfile(finfo, fraw, inputfile, taginfo):
    nwarning = 0
    lack_labels = set()
    for index, line in enumerate(open(inputfile, 'r')):
        line = unicode(line, 'utf-8')
        if index == 0:
            continue
        tokens = line.strip().replace('/', ',').split(',')
        tokens = filter(lambda x: x != '', tokens)
        tokens = map(int, tokens)
        tokens = list(set(tokens))
        if FLAGS.sort_tags:
            tokens.sort()
        if len(tokens) < FLAGS.min_labels:
            continue
        for tag in tokens:
            if tag not in taginfo:
                nwarning += 1
                lack_labels.add(tag)
                finfo.write(str(tag))
            else:
                finfo.write(taginfo[tag].encode('utf-8'))
            fraw.write(str(tag))
            finfo.write(' ')
            fraw.write(' ')
        finfo.write('\n')
        fraw.write('\n')
        if index % 500000 == 0:
            print("{}: {} lines processed".format(inputfile, index))

    # print("lack info of labels [{}]: {}".format(nwarning, lack_labels))


def convert():
    taginfo = load_tag_info_dict()
    with open(FLAGS.output_info, 'w') as finfo:
        with open(FLAGS.output_raw, 'w') as fraw:
            if FLAGS.input_video_tags_file != '':
                convertfile(finfo, fraw, FLAGS.input_video_tags_file, taginfo)
            if FLAGS.input_article_tags_file != '':
                convertfile(finfo, fraw,
                            FLAGS.input_article_tags_file, taginfo)


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
        '--output_info',
        type=str,
        default='',
        help='Output fasttext format, tag id converts to tag info.'
    )

    parser.add_argument(
        '--output_raw',
        type=str,
        default='',
        help='Output fasttext format, tag id unchanged.'
    )

    parser.add_argument(
        '--min_labels',
        type=int,
        default=1,
        help=''
    )

    parser.add_argument(
        '--sort_tags',
        type=bool,
        default=True,
        help=''
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
