#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from options import Options

import os
import json
import model_keys
import argparse


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'


parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', default='', type=str, help='')
parser.add_argument('--eval_data_path', default='', type=str, help='')
parser.add_argument('--lr', default=0.25, type=float, help='learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--max_train_steps', default=None, type=int, help='')
parser.add_argument('--epoch', default=1, type=int, help='')
parser.add_argument('--model_dir', default="model_dir", type=str, help='')
parser.add_argument('--export_model_dir', default="export_model_dir",
                    type=str, help='')
parser.add_argument('--prefetch_size', default=10000, type=int, help='')
parser.add_argument('--shuffle_size', default=10000, type=int, help='')
parser.add_argument('--save_summary_steps', default=100, type=int, help='')
parser.add_argument('--save_checkpoints_secs', default=600, type=int, help='')
parser.add_argument('--keep_checkpoint_max', default=3, type=int, help='')
parser.add_argument('--log_step_count_steps', default=100, type=int, help='')
parser.add_argument('--use_profile_hook', default=0, type=int, help='')
parser.add_argument('--profile_steps', default=100, type=int, help='')
parser.add_argument('--remove_model_dir', default=1, type=int, help='')
parser.add_argument('--dropout', default=0.1, type=float, help='')
parser.add_argument(
    '--max_distribute_train_steps', default=None, type=int, help='')
parser.add_argument('--shuffle_batch', default=0, type=int, help='')
parser.add_argument('--optimizer_type', default='ada', type=str, help='')
parser.add_argument('--tfrecord_file', default='', type=str, help='')
parser.add_argument('--num_tfrecord_file', default=1, type=int, help='')
parser.add_argument(
    '--train_data_format', default='fasttext', type=str, help='')
parser.add_argument(
    '--map_num_parallel_calls', default=1, type=int, help='')
parser.add_argument(
    '--train_parallel_mode', default='default', type=str, help='')
parser.add_argument('--num_parallel', default=1, type=int, help='')
parser.add_argument(
    '--sgd_lr_decay_type', default='exponential_decay', type=str, help='')
parser.add_argument('--sgd_lr_decay_steps', default=100, type=int, help='')
parser.add_argument('--sgd_lr_decay_rate', default=0.99, type=float, help='')
parser.add_argument('--use_clip_gradients', default=0, type=int, help='')
parser.add_argument('--clip_norm', default=5.0, type=float, help='')
parser.add_argument('--sgd_lr_decay_end_learning_rate',
                    default=0.0001, type=float, help='')
parser.add_argument('--sgd_lr_decay_power', default=1.0, type=float, help='')

opts = Options()


def parse_args(argv):
    args = parser.parse_args(argv[1:])
    opts.train_data_path = args.train_data_path
    opts.eval_data_path = args.eval_data_path
    opts.lr = args.lr
    opts.batch_size = args.batch_size
    opts.max_train_steps = args.max_train_steps
    if opts.max_train_steps is not None and opts.max_train_steps < 0:
        opts.max_train_steps = None
    opts.epoch = args.epoch
    opts.model_dir = args.model_dir
    opts.export_model_dir = args.export_model_dir
    opts.prefetch_size = args.prefetch_size
    opts.shuffle_size = args.shuffle_size
    opts.save_summary_steps = args.save_summary_steps
    if opts.save_summary_steps < 0:
        opts.save_summary_steps = None
    opts.save_checkpoints_secs = args.save_checkpoints_secs
    opts.keep_checkpoint_max = args.keep_checkpoint_max
    opts.log_step_count_steps = args.log_step_count_steps
    opts.use_profile_hook = bool(args.use_profile_hook)
    opts.profile_steps = args.profile_steps
    opts.remove_model_dir = bool(args.remove_model_dir)
    opts.dropout = args.dropout
    opts.max_distribute_train_steps = args.max_distribute_train_steps
    if (opts.max_distribute_train_steps is not None
            and opts.max_distribute_train_steps < 0):
        opts.max_distribute_train_steps = None
    opts.shuffle_batch = bool(args.shuffle_batch)
    opts.predict_ws = args.predict_ws
    opts.optimizer_type = args.optimizer_type
    opts.tfrecord_file = args.tfrecord_file
    opts.num_tfrecord_file = args.num_tfrecord_file
    opts.train_data_format = args.train_data_format
    opts.map_num_parallel_calls = args.map_num_parallel_calls
    opts.train_parallel_mode = args.train_parallel_mode
    opts.num_parallel = args.num_parallel
    opts.sgd_lr_decay_type = args.sgd_lr_decay_type
    opts.sgd_lr_decay_steps = args.sgd_lr_decay_steps
    opts.sgd_lr_decay_rate = args.sgd_lr_decay_rate
    opts.use_clip_gradients = bool(args.use_clip_gradients)
    opts.clip_norm = args.clip_norm
    opts.sgd_lr_decay_end_learning_rate = args.sgd_lr_decay_end_learning_rate
    opts.sgd_lr_decay_power = args.sgd_lr_decay_power


def validate_opts():
    if opts.dropout < 0.0:
        raise ValueError("dropout should not less than 0")

    if (opts.tf_config is not None and
            opts.train_parallel_mode != model_keys.TrainParallelMode.DEFAULT):
        raise ValueError(
            "Distribute mode only surpport 'default' train parallel mode.")


def parse_tf_config():
    """Parse environment TF_CONFIG. config put in opts."""

    tf_config = os.environ.get('TF_CONFIG')
    if tf_config is not None:
        tf_config = json.loads(tf_config)

    opts.tf_config = tf_config
    opts.task_type = model_keys.TaskType.LOCAL  # default mode
    if opts.tf_config is not None:
        opts.task_type = opts.tf_config['task']['type']
        opts.task_index = opts.tf_config['task']['index']


def parse(argv):
    parse_args(argv)
    parse_tf_config()
    validate_opts()

    return opts
