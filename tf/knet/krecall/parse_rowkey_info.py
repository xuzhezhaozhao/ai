#! /usr/bin/env python
# -*- coding=utf8 -*-

import json
import collections
import tensorflow as tf


RowkeyInfo = collections.namedtuple(
    'RowkeyInfo', ['rowkey', 'screen_type', 'cover_score',
                   'duration', 'play', 'e_play', 'exposure', 'push_time'])


def parse_rowkey_info(rowkey_info_file):
    """Parse rowkey info
    Return: Python dict, rowkey => RowkeyInfo
    """

    tf.logging.info("parse rowkey info ...")

    D = dict()
    for lineindex, line in enumerate(open(rowkey_info_file)):
        if lineindex % 100000 == 0:
            tf.logging.info("parse {}w lines ...".format(lineindex / 10000))
        line = line.strip()
        try:
            root = json.loads(line)
            rowkey = root['rowkey']
            screen_type = root.get('short_v', 0)
            cover_score = root.get('cover_score', 0)
            duration = root.get('duration', 0)
            play = root.get('click', 0)  # history problem, click is play
            e_play = root.get('e_click', 0)
            exposure = root.get('exposure', 0)
            push_time = root.get('push_time', 0)

            rowkey_info = RowkeyInfo(
                rowkey=rowkey, screen_type=screen_type,
                cover_score=cover_score, duration=duration, play=play,
                e_play=e_play, exposure=exposure, push_time=push_time)

            D[rowkey] = rowkey_info
        except Exception as e:
            print("catch exception in line {}, exception: {}, line = {}"
                  .format(lineindex, e, line))

    tf.logging.info("parse rowkey info done, size = {}.".format(len(D)))

    return D
