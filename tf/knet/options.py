#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Options(object):
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
        self.min_count_label = 1
        self.label = "__label__"
        self.batch_size = 1
        self.num_sampled = 5
        self.max_train_steps = None
        self.epoch = 1
        self.hidden_units = []
        self.model_dir = 'model_dir'
        self.export_model_dir = 'export_model_dir'
        self.prefetch_size = 1000

        self.save_summary_steps = 100
        self.save_checkpoints_secs = 600
        self.keep_checkpoint_max = 3
        self.log_step_count_steps = 100

        self.recall_k = 10
        self.dict_dir = 'dict_dir'
        self.use_saved_dict = False

    def __str__(self):
        return \
            "Option:\n \
        train_data_path = {} \n \
        lr = {} \n \
        dim = {} \n \
        maxn = {} \n \
        minn = {} \n \
        word_ngrams = {} \n \
        bucket = {} \n \
        ws = {} \n \
        min_count = {} \n \
        t = {} \n \
        verbose = {} \n \
        min_count_label = {} \n \
        label = {} \n \
        batch_size = {} \n \
        num_sampled = {} \n \
        max_train_steps = {} \n \
        epoch = {} \n \
        hidden_units = {} \n \
        model_dir = {} \n \
        export_model_dir = {} \n \
        prefetch_size = {} \n \
        save_summary_steps = {} \n \
        save_checkpoints_secs = {} \n \
        keep_checkpoint_max = {} \n \
        log_step_count_steps = {} \n \
        recall_k = {} \n \
        dict_dir = {} \n \
        use_saved_dict = {} \n \
        ".format(
                self.train_data_path,
                self.lr,
                self.dim,
                self.maxn,
                self.minn,
                self.word_ngrams,
                self.bucket,
                self.ws,
                self.min_count,
                self.t,
                self.verbose,
                self.min_count_label,
                self.label,
                self.batch_size,
                self.num_sampled,
                self.max_train_steps,
                self.epoch,
                self.hidden_units,
                self.model_dir,
                self.export_model_dir,
                self.prefetch_size,
                self.save_summary_steps,
                self.save_checkpoints_secs,
                self.keep_checkpoint_max,
                self.log_step_count_steps,
                self.recall_k,
                self.dict_dir,
                self.use_saved_dict
            )
