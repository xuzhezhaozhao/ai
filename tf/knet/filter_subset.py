#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import input_data


def filter_subset(dict_dir):
    save_weights_path = os.path.join(dict_dir, SAVE_NCE_WEIGHTS_NAME)
    save_biases_path = os.path.join(dict_dir, SAVE_NCE_BIASES_NAME)
    nce_weights = np.load(save_weights_path)
    nce_biases = np.load(save_biases_path)

    dict_words_path = os.path.join(dict_dir, input_data.DICT_WORDS)
    words = [line.strip() for line in open(dict_words_path)
             if line.strip() != '']
