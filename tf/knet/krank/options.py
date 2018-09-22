#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Options(object):
    def __init__(self):
        # cmd args
        self.train_data_path = ""
        self.eval_data_path = ""
        self.feature_manager_path = ""
        self.lr = 0.05
        self.rowkey_embedding_dim = 100
        self.train_ws = 5
        self.batch_size = 64
        self.max_train_steps = None
        self.epoch = 1
        self.hidden_units = []
        self.model_dir = 'model_dir'
        self.export_model_dir = 'export_model_dir'
        self.prefetch_size = 1000
        self.shuffle_batch = True
        self.shuffle_size = 1000
        self.save_summary_steps = 100
        self.save_checkpoints_secs = 600
        self.keep_checkpoint_max = 3
        self.log_step_count_steps = 100
        self.use_profile_hook = False
        self.profile_steps = 100
        self.remove_model_dir = 1
        self.dropout = 0.3
        self.map_num_parallel_calls = 1
        self.rowkey_dict_path = ''
        self.num_rowkey = 0
        self.inference_actions_len = 5

        # non-cmd args
        self.estimator = None
        self.hooks = None

    def __str__(self):
        return \
            "Option:\n \
        train_data_path = {} \n \
        eval_data_path = {} \n \
        feature_manager_path = {} \n \
        lr = {} \n \
        rowkey_embedding_dim = {} \n \
        train_ws = {} \n \
        batch_size = {} \n \
        max_train_steps = {} \n \
        epoch = {} \n \
        hidden_units = {} \n \
        model_dir = {} \n \
        export_model_dir = {} \n \
        prefetch_size = {} \n \
        shuffle_batch = {} \n \
        shuffle_size = {} \n \
        save_summary_steps = {} \n \
        save_checkpoints_secs = {} \n \
        keep_checkpoint_max = {} \n \
        log_step_count_steps = {} \n \
        use_profile_hook = {} \n \
        profile_steps = {} \n \
        remove_model_dir = {} \n \
        dropout = {} \n \
        map_num_parallel_calls = {} \n \
        rowkey_dict_path = {} \n \
        inference_actions_len = {} \n \
        num_rowkey = {} \n \
        estimator = {} \n \
        hooks = {} \n \
        ".format(
                self.train_data_path,
                self.eval_data_path,
                self.feature_manager_path,
                self.lr,
                self.rowkey_embedding_dim,
                self.train_ws,
                self.batch_size,
                self.max_train_steps,
                self.epoch,
                self.hidden_units,
                self.model_dir,
                self.export_model_dir,
                self.prefetch_size,
                self.shuffle_batch,
                self.shuffle_size,
                self.save_summary_steps,
                self.save_checkpoints_secs,
                self.keep_checkpoint_max,
                self.log_step_count_steps,
                self.use_profile_hook,
                self.profile_steps,
                self.remove_model_dir,
                self.dropout,
                self.map_num_parallel_calls,
                self.rowkey_dict_path,
                self.inference_actions_len,
                self.num_rowkey,

                # non-cmd args
                self.estimator,
                self.hooks,
            )
