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
parser.add_argument('--dim', default=100, type=int, help='embedding dim')
parser.add_argument('--train_ws', default=5, type=int, help='window size')
parser.add_argument('--min_count', default=50, type=int, help='')
parser.add_argument('--t', default=0.0001, type=float, help='')
parser.add_argument('--verbose', default=1, type=int, help='')
parser.add_argument('--min_count_label', default=1, type=int, help='')
parser.add_argument('--label', default="__label__", type=str, help='')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--num_sampled', default=5, type=int, help='')
parser.add_argument('--max_train_steps', default=None, type=int, help='')
parser.add_argument('--epoch', default=1, type=int, help='')
parser.add_argument('--hidden_units', default="64,64", type=str, help='')
parser.add_argument('--model_dir', default="model_dir", type=str, help='')
parser.add_argument('--export_model_dir', default="export_model_dir",
                    type=str, help='')
parser.add_argument('--prefetch_size', default=10000, type=int, help='')
parser.add_argument('--save_summary_steps', default=100, type=int, help='')
parser.add_argument('--save_checkpoints_secs', default=600, type=int, help='')
parser.add_argument('--keep_checkpoint_max', default=3, type=int, help='')
parser.add_argument('--log_step_count_steps', default=100, type=int, help='')
parser.add_argument('--recall_k', default=1, type=int, help='')
parser.add_argument('--dict_dir', default="dict_dir", type=str, help='')
parser.add_argument('--use_saved_dict', default=0, type=int, help='')
parser.add_argument('--use_profile_hook', default=0, type=int, help='')
parser.add_argument('--profile_steps', default=100, type=int, help='')
parser.add_argument('--root_ops_path', default='', type=str, help='')
parser.add_argument('--remove_model_dir', default=1, type=int, help='')
parser.add_argument('--optimize_level', default=1, type=int, help='')
parser.add_argument('--receive_ws', default=5, type=int, help='')
parser.add_argument('--use_subset', default=0, type=int, help='')
parser.add_argument('--dropout', default=0.1, type=float, help='')
parser.add_argument('--ntargets', default=1, type=int, help='')
parser.add_argument('--chief_lock', default='chief.lock', type=str, help='')
parser.add_argument(
    '--max_distribute_train_steps', default=None, type=int, help='')
parser.add_argument('--train_nce_biases', default=0, type=int, help='')
parser.add_argument('--shuffle_batch', default=0, type=int, help='')
parser.add_argument('--predict_ws', default=5, type=int, help='')
parser.add_argument('--sample_dropout', default=0.5, type=float, help='')
parser.add_argument('--optimizer_type', default='ada', type=str, help='')
parser.add_argument('--num_in_graph_replication', default=1, type=int, help='')
parser.add_argument('--threads', default=1, type=int, help='')

opts = Options()


def parse_args(argv):
    args = parser.parse_args(argv[1:])
    opts.train_data_path = args.train_data_path
    opts.eval_data_path = args.eval_data_path
    opts.lr = args.lr
    opts.dim = args.dim
    opts.train_ws = args.train_ws
    opts.min_count = args.min_count
    opts.t = args.t
    opts.verbose = args.verbose
    opts.min_count_label = args.min_count_label
    opts.label = args.label
    opts.batch_size = args.batch_size
    opts.num_sampled = args.num_sampled
    opts.max_train_steps = args.max_train_steps
    if opts.max_train_steps is not None and opts.max_train_steps < 0:
        opts.max_train_steps = None
    opts.epoch = args.epoch
    opts.hidden_units = map(int, filter(lambda x: x != '',
                                        args.hidden_units.split(',')))
    opts.model_dir = args.model_dir
    opts.export_model_dir = args.export_model_dir
    opts.prefetch_size = args.prefetch_size
    opts.save_summary_steps = args.save_summary_steps
    opts.save_checkpoints_secs = args.save_checkpoints_secs
    opts.keep_checkpoint_max = args.keep_checkpoint_max
    opts.log_step_count_steps = args.log_step_count_steps
    opts.recall_k = args.recall_k
    opts.dict_dir = args.dict_dir
    opts.use_saved_dict = bool(args.use_saved_dict)
    opts.use_profile_hook = bool(args.use_profile_hook)
    opts.profile_steps = args.profile_steps
    opts.root_ops_path = args.root_ops_path
    opts.remove_model_dir = bool(args.remove_model_dir)
    opts.optimize_level = args.optimize_level
    opts.receive_ws = args.receive_ws
    opts.use_subset = bool(args.use_subset)
    opts.dropout = args.dropout
    opts.ntargets = args.ntargets
    opts.chief_lock = args.chief_lock
    opts.max_distribute_train_steps = args.max_distribute_train_steps
    if (opts.max_distribute_train_steps is not None
            and opts.max_distribute_train_steps < 0):
        opts.max_distribute_train_steps = None
    opts.train_nce_biases = bool(args.train_nce_biases)
    opts.shuffle_batch = bool(args.shuffle_batch)
    opts.predict_ws = args.predict_ws
    opts.sample_dropout = args.sample_dropout
    opts.optimizer_type = args.optimizer_type
    opts.num_in_graph_replication = args.num_in_graph_replication
    opts.threads = args.threads


def validate_opts():
    if opts.optimize_level not in model_keys.ALL_OPTIMIZE_LEVELS:
        raise ValueError(
            "optimaize_level {} not surpported.".format(opts.optimize_level))
    if opts.predict_ws > opts.receive_ws:
        raise ValueError(
            "predict_ws[{}] should not be larger than receive_ws[{}]".format(
                opts.predict_ws, opts.receive_ws))
    if len([u for u in opts.hidden_units if u <= 0]) > 0:
        raise ValueError("hidden_units contain unit <= 0")
    if opts.dropout < 0.0:
        raise ValueError("dropout should not less than 0")
    if opts.num_in_graph_replication < 1:
        raise ValueError("num_in_graph_replication should not less than 1")


def parse_tf_config():
    """Parse environment TF_CONFIG. config put in opts."""

    try:
        tf_config = os.environ.get('TF_CONFIG')
        tf_config = json.loads(tf_config)
    except Exception:
        tf_config = None

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
