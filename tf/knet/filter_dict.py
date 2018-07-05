#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np

import model_keys


D = None  # rowkey_info dict


def filter_and_save_subset(opts):
    dict_dir = opts.dict_dir
    save_weights_path = os.path.join(
        dict_dir, model_keys.SAVE_NCE_WEIGHTS_NAME)
    save_biases_path = os.path.join(dict_dir, model_keys.SAVE_NCE_BIASES_NAME)
    nce_weights = np.load(save_weights_path)
    nce_biases = np.load(save_biases_path)
    dict_words_path = os.path.join(dict_dir, model_keys.DICT_WORDS)
    words = [line.strip() for line in open(dict_words_path)
             if line.strip() != '']

    subset_words = [w for w in words if filter_rowkey(w, opts)]
    num_in_subset = len(subset_words)
    assert num_in_subset > 0, 'subset is empty.'
    dim = nce_weights.shape[-1]

    subset_weights = np.zeros([num_in_subset, dim], dtype=np.float32)
    subset_biases = np.zeros([num_in_subset], dtype=np.float32)
    tf.logging.info("subset length = {}".format(num_in_subset))

    subset_index = 0
    for index, word in enumerate(words):
        if filter_rowkey(word, opts):
            # index plus one because of padding
            subset_weights[subset_index] = nce_weights[index + 1]
            subset_biases[subset_index] = nce_biases[index + 1]
            subset_index += 1

    to_save_subset_words = reduce(lambda w1, w2: w1 + '\n' + w2, subset_words)

    with open(os.path.join(dict_dir, model_keys.DICT_WORDS_SUBSET), 'w') as f:
        f.write(to_save_subset_words)

    np.save(os.path.join(dict_dir, model_keys.SAVE_NCE_WEIGHTS_SUBSET_NAME),
            subset_weights)
    np.save(os.path.join(dict_dir, model_keys.SAVE_NCE_BIASES_SUBSET_NAME),
            subset_biases)


def filter_rowkey(rowkey, opts):
    """Filter rowkey."""

    if opts.filter_with_rowkey_info:
        return filter_with_rowkey_info(rowkey, opts)
    else:
        return filter_video_rowkey(rowkey)


def filter_video_rowkey(rowkey):
    """Check wheather rowkey is video, yes return True, or False."""

    if len(rowkey) > 5:
        if rowkey[-2] == 'a' or rowkey[-2] == 'b':
            return True
    return False


def filter_with_rowkey_info(rowkey, opts):
    import parse_rowkey_info

    global D

    if D is None:
        D = parse_rowkey_info.parse_rowkey_info(opts.rowkey_info_file)

    if rowkey not in D:
        return False

    info = D[rowkey]

    if info.exposure < opts.filter_with_rowkey_info_exposure_thr:
        return False
    if info.play < opts.filter_with_rowkey_info_play:
        return False
    if info.e_play < opts.filter_with_rowkey_info_e_play:
        return False

    e_play_ratio = 0.0
    if info.play > 0:
        e_play_ratio = float(info.e_play) / info.play
    if e_play_ratio < opts.filter_with_rowkey_info_e_play_ratio_thr:
        return False

    return True
