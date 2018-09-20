#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from options import Options

import os
import argparse


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'


parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', default='', type=str, help='')
parser.add_argument('--eval_data_path', default='', type=str, help='')
parser.add_argument('--feature_manager_path', default='', type=str, help='')
parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
parser.add_argument('--rowkey_embedding_dim', default=100, type=int, help='')
parser.add_argument('--train_ws', default=5, type=int, help='')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--max_train_steps', default=None, type=int, help='')
parser.add_argument('--epoch', default=1, type=int, help='')
parser.add_argument('--hidden_units', default="64,64", type=str, help='')
parser.add_argument('--model_dir', default="model_dir", type=str, help='')
parser.add_argument('--export_model_dir', default="export_model_dir",
                    type=str, help='')
parser.add_argument('--prefetch_size', default=10000, type=int, help='')
parser.add_argument('--shuffle_batch', default=1, type=int, help='')
parser.add_argument('--shuffle_size', default=10000, type=int, help='')
parser.add_argument('--save_summary_steps', default=100, type=int, help='')
parser.add_argument('--save_checkpoints_secs', default=600, type=int, help='')
parser.add_argument('--keep_checkpoint_max', default=3, type=int, help='')
parser.add_argument('--log_step_count_steps', default=100, type=int, help='')
parser.add_argument('--use_profile_hook', default=0, type=int, help='')
parser.add_argument('--profile_steps', default=100, type=int, help='')
parser.add_argument('--remove_model_dir', default=1, type=int, help='')
parser.add_argument('--dropout', default=0.3, type=float, help='')
parser.add_argument('--map_num_parallel_calls', default=1, type=int, help='')
parser.add_argument('--rowkey_dict_path', default='', type=str, help='')

opts = Options()


def parse_args(argv):
    args = parser.parse_args(argv[1:])
    opts.train_data_path = args.train_data_path
    opts.eval_data_path = args.eval_data_path
    opts.feature_manager_path = args.feature_manager_path
    opts.lr = args.lr
    opts.rowkey_embedding_dim = args.rowkey_embedding_dim
    opts.train_ws = args.train_ws
    opts.batch_size = args.batch_size
    opts.max_train_steps = args.max_train_steps
    if opts.max_train_steps is not None and opts.max_train_steps < 0:
        opts.max_train_steps = None

    opts.epoch = args.epoch
    opts.hidden_units = map(int, filter(lambda x: x != '',
                                        args.hidden_units.split(',')))
    opts.model_dir = args.model_dir
    opts.export_model_dir = args.export_model_dir
    opts.prefetch_size = args.prefetch_size
    opts.shuffle_batch = bool(args.shuffle_batch)
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
    opts.map_num_parallel_calls = args.map_num_parallel_calls
    opts.rowkey_dict_path = args.rowkey_dict_path
    opts.num_rowkey = 1 + len([line for line in open(opts.rowkey_dict_path)
                               if line != ''])  # plus one for padding


def validate_opts():
    if len([u for u in opts.hidden_units if u <= 0]) > 0:
        raise ValueError("hidden_units contain unit <= 0")
    if opts.dropout < 0.0:
        raise ValueError("dropout should not less than 0")


def parse(argv):
    parse_args(argv)
    validate_opts()

    return opts