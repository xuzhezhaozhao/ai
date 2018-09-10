#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import model_keys


class Options(object):
    def __init__(self):
        # cmd args
        self.train_data_path = ""
        self.eval_data_path = ""
        self.lr = 0.05
        self.batch_size = 1
        self.max_train_steps = None
        self.epoch = 1
        self.model_dir = 'model_dir'
        self.export_model_dir = 'export_model_dir'
        self.prefetch_size = 1000
        self.shuffle_size = 1000
        self.save_summary_steps = 100
        self.save_checkpoints_secs = 600
        self.keep_checkpoint_max = 3
        self.log_step_count_steps = 100
        self.use_profile_hook = False
        self.profile_steps = 100
        self.remove_model_dir = 1
        self.dropout = 0.1
        self.shuffle_batch = False
        self.optimizer_type = 'ada'
        self.tfrecord_file = ''
        self.num_tfrecord_file = 1
        self.train_data_format = 'fasttext'
        self.map_num_parallel_calls = 1
        self.train_parallel_mode = 'default'
        self.num_parallel = 1
        self.sgd_lr_decay_type = 'exponential_decay'
        self.sgd_lr_decay_steps = 100
        self.sgd_lr_decay_rate = 0.99
        self.use_clip_gradients = 0
        self.clip_norm = 5.0
        self.sgd_lr_decay_end_learning_rate = 0.0001
        self.sgd_lr_decay_power = 1.0
        self.num_classes = 1
        self.pretrained_weights_path = ''
        self.train_layers = []

        # non-cmd args
        self.estimator = None
        self.hooks = None
        self.tf_config = None
        self.task_type = model_keys.TaskType.LOCAL
        self.task_index = None

    def __str__(self):
        return \
            "Option:\n \
        train_data_path = {} \n \
        eval_data_path = {} \n \
        lr = {} \n \
        batch_size = {} \n \
        max_train_steps = {} \n \
        epoch = {} \n \
        model_dir = {} \n \
        export_model_dir = {} \n \
        prefetch_size = {} \n \
        shuffle_size = {} \n \
        save_summary_steps = {} \n \
        save_checkpoints_secs = {} \n \
        keep_checkpoint_max = {} \n \
        log_step_count_steps = {} \n \
        use_profile_hook = {} \n \
        profile_steps = {} \n \
        remove_model_dir = {} \n \
        dropout = {} \n \
        shuffle_batch = {} \n \
        optimizer_type = {} \n \
        tfrecord_file = {} \n \
        num_tfrecord_file = {} \n \
        train_data_format = {} \n \
        map_num_parallel_calls = {} \n \
        train_parallel_mode = {} \n \
        num_parallel = {} \n \
        sgd_lr_decay_type = {} \n \
        sgd_lr_decay_steps = {} \n \
        sgd_lr_decay_rate = {} \n \
        use_clip_gradients = {} \n \
        clip_norm = {} \n \
        sgd_lr_decay_end_learning_rate = {} \n \
        sgd_lr_decay_power = {} \n \
        num_classes = {} \n \
        pretrained_weights_path = {} \n \
        train_layers = {} \n \
        \n \
        estimator = {} \n \
        hooks = {} \n \
        tf_config = {} \n \
        task_type = {} \n \
        task_index = {} \n \
        ".format(
                self.train_data_path,
                self.eval_data_path,
                self.lr,
                self.batch_size,
                self.max_train_steps,
                self.epoch,
                self.model_dir,
                self.export_model_dir,
                self.prefetch_size,
                self.shuffle_size,
                self.save_summary_steps,
                self.save_checkpoints_secs,
                self.keep_checkpoint_max,
                self.log_step_count_steps,
                self.use_profile_hook,
                self.profile_steps,
                self.remove_model_dir,
                self.dropout,
                self.shuffle_batch,
                self.optimizer_type,
                self.tfrecord_file,
                self.num_tfrecord_file,
                self.train_data_format,
                self.map_num_parallel_calls,
                self.train_parallel_mode,
                self.num_parallel,
                self.sgd_lr_decay_type,
                self.sgd_lr_decay_steps,
                self.sgd_lr_decay_rate,
                self.use_clip_gradients,
                self.clip_norm,
                self.sgd_lr_decay_end_learning_rate,
                self.sgd_lr_decay_power,
                self.num_classes,
                self.pretrained_weights_path,
                self.train_layers,
                # non-cmd args
                self.estimator,
                self.hooks,
                self.tf_config,
                self.task_type,
                self.task_index)
