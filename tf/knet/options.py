#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Options(object):
    """Options used by our knet model"""

    def __init__(self):
        self.train_data_path = ""
        self.lr = 0.05
        self.dim = 100
        self.maxn = 0
        self.minn = 0
        self.word_ngrams = 1
        self.bucket = 2000000
        self.ws = 5
        self.min_count = 5
        self.t = 0.0001
        self.verbose = 1
        self.min_count_label = 5
        self.label = "__label__"
        self.batch_size = 1
        self.seed = 1
        self.num_sampled = 5
        self.train_steps = 1000
