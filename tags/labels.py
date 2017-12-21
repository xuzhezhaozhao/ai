#! /usr/bin/env python
# -*-coding:utf-8 -*-

import argparse


# Basic model parameters as external flags.
FLAGS = None


def delteduplicated(iterable):
    uniq = list()
    for x in iterable:
        if x not in uniq:
            uniq.append(x)
    return uniq


def load_tag_info_dict(inputfile):
    taginfo = {}
    for index, line in enumerate(open(inputfile, 'r')):
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


def convertfile(finfo, inputfile, taginfo, labeldict1, labeldict2):
    lack_labels = set()

    for index, line in enumerate(open(inputfile, 'r')):
        line = unicode(line, 'utf-8')
        if index == 0:
            continue
        tokens = line.strip().replace('/', ',').split(',')
        tagids = tokens[1:-4]
        tagids = filter(lambda x: x != '', tagids)
        tagids = map(int, tagids)
        tagids = delteduplicated(tagids)
        if FLAGS.sort_tags:
            tagids.sort()

        class1_name = tokens[-3]
        class2_name = tokens[-1]

        if len(tagids) < FLAGS.min_tags:
            continue

        if class1_name != "":
            class1_id = int(tokens[-4])
            # finfo.write("__label__" + class1_name.encode('utf-8') + " ")
            labeldict1[class1_name] = class1_id

        if class2_name != "":
            class2_id = int(tokens[-2])
            finfo.write("__label__" + class2_name.encode('utf-8') + " ")
            labeldict2[class2_name] = class2_id

        if class1_name == "" and class2_name == "":
            continue

        for tag in tagids:
            if tag not in taginfo:
                lack_labels.add(tag)
                finfo.write(str(tag))
            else:
                finfo.write(taginfo[tag].encode('utf-8'))
            finfo.write(' ')
        finfo.write('\n')
        if index % 500000 == 0:
            print("{}: {} lines processed".format(inputfile, index))


def convert():
    taginfo = load_tag_info_dict(FLAGS.input_tag_info_file)
    labeldict1 = dict()
    labeldict2 = dict()

    with open(FLAGS.output_info, 'w') as finfo:
        if FLAGS.input_video_tags_file != '':
            convertfile(finfo, FLAGS.input_video_tags_file, taginfo,
                        labeldict1, labeldict2)
        if FLAGS.input_article_tags_file != '':
            convertfile(finfo, FLAGS.input_article_tags_file, taginfo,
                        labeldict1, labeldict2)

    print("write label dict ...")
    with open(FLAGS.output_label_dict_file, 'w') as f:
        for k in labeldict1:
            f.write(str(labeldict1[k]) + ' ')
            f.write(k.encode('utf-8') + ' 1')
            f.write('\n')
        for k in labeldict2:
            f.write(str(labeldict2[k]) + ' ')
            f.write(k.encode('utf-8') + ' 2')
            f.write('\n')


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
        '--min_tags',
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

    parser.add_argument(
        '--output_label_dict_file',
        type=str,
        default="",
        help=''
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
