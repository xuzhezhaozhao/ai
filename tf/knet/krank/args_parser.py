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
parser.add_argument('--batch_size', default=64, type=int, help='')
parser.add_argument('--eval_batch_size', default=64, type=int, help='')
parser.add_argument('--max_train_steps', default=None, type=int, help='')
parser.add_argument('--max_eval_steps', default=None, type=int, help='')
parser.add_argument('--max_eval_steps_on_train_dataset',
                    default=1000, type=int, help='')
parser.add_argument('--epoch', default=1, type=int, help='')
parser.add_argument('--hidden_units', default="64,64", type=str, help='')
parser.add_argument('--model_dir', default="model_dir", type=str, help='')
parser.add_argument('--export_model_dir', default="export_model_dir",
                    type=str, help='')
parser.add_argument('--prefetch_size', default=10000, type=int, help='')
parser.add_argument('--shuffle_batch', default=1, type=int, help='')
parser.add_argument('--shuffle_size', default=10000, type=int, help='')
parser.add_argument('--save_summary_steps', default=100, type=int, help='')
parser.add_argument('--save_checkpoints_secs', default=100, type=int, help='')
parser.add_argument('--keep_checkpoint_max', default=3, type=int, help='')
parser.add_argument('--log_step_count_steps', default=100, type=int, help='')
parser.add_argument('--log_step_count_secs', default=10, type=int, help='')
parser.add_argument('--remove_model_dir', default=1, type=int, help='')
parser.add_argument('--dropout', default=0.3, type=float, help='')
parser.add_argument('--map_num_parallel_calls', default=1, type=int, help='')
parser.add_argument('--rowkey_dict_path', default='', type=str, help='')
parser.add_argument('--inference_actions_len', default=5, type=int, help='')
parser.add_argument('--inference_num_targets', default=100, type=int, help='')
parser.add_argument('--train_parallel_mode',
                    default='default', type=str, help='')
parser.add_argument('--train_num_parallel', default=1, type=int, help='')
parser.add_argument('--optimizer_type', default='sgd', type=str, help='')
parser.add_argument('--optimizer_epsilon', default=1e-8, type=float, help='')
parser.add_argument('--optimizer_adadelta_rho',
                    default=0.95, type=float, help='')
parser.add_argument('--optimizer_adam_beta1',
                    default=0.9, type=float, help='')
parser.add_argument('--optimizer_adam_beta2',
                    default=0.999, type=float, help='')
parser.add_argument('--optimizer_rmsprop_decay',
                    default=0.9, type=float, help='')
parser.add_argument('--optimizer_rmsprop_momentum',
                    default=0.0, type=float, help='')
parser.add_argument('--optimizer_rmsprop_centered',
                    default=0, type=int, help='')
parser.add_argument('--optimizer_momentum_momentum',
                    default=0.9, type=float, help='')
parser.add_argument('--optimizer_momentum_use_nesterov',
                    default=0, type=int, help='')
parser.add_argument('--clip_gradients', default=0, type=int, help='')
parser.add_argument('--clip_gradients_norm', default=5.0, type=float, help='')
parser.add_argument('--l2_regularizer', default=0.0001, type=float, help='')
parser.add_argument('--use_early_stopping', default=0, type=int, help='')
parser.add_argument('--auc_num_thresholds', default=200, type=int, help='')
parser.add_argument('--optimizer_exponential_decay_steps',
                    default=10000, type=int, help='')
parser.add_argument('--optimizer_exponential_decay_rate',
                    default=0.96, type=float, help='')
parser.add_argument('--optimizer_exponential_decay_staircase',
                    default=0, type=int, help='')
parser.add_argument('--optimizer_ftrl_lr_power',
                    default=-0.5, type=float, help='')
parser.add_argument('--optimizer_ftrl_initial_accumulator_value',
                    default=0.1, type=float, help='')
parser.add_argument('--optimizer_ftrl_l1_regularization',
                    default=0.0, type=float, help='')
parser.add_argument('--optimizer_ftrl_l2_regularization',
                    default=0.0, type=float, help='')
parser.add_argument('--optimizer_ftrl_l2_shrinkage_regularization',
                    default=0.0, type=float, help='')
parser.add_argument('--evaluate_every_secs', default=30, type=int, help='')
parser.add_argument('--leaky_relu_alpha', default=0.3, type=float, help='')

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
    opts.eval_batch_size = args.eval_batch_size
    opts.max_train_steps = args.max_train_steps
    if opts.max_train_steps is not None and opts.max_train_steps <= 0:
        opts.max_train_steps = None

    opts.max_eval_steps = args.max_eval_steps
    if opts.max_eval_steps is not None and opts.max_eval_steps <= 0:
        opts.max_eval_steps = None

    opts.max_eval_steps_on_train_dataset = args.max_eval_steps_on_train_dataset
    if (opts.max_eval_steps_on_train_dataset is not None
            and opts.max_eval_steps_on_train_dataset <= 0):
        opts.max_eval_steps_on_train_dataset = None

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
    opts.log_step_count_secs = args.log_step_count_secs
    opts.remove_model_dir = bool(args.remove_model_dir)
    opts.dropout = args.dropout
    opts.map_num_parallel_calls = args.map_num_parallel_calls
    opts.rowkey_dict_path = args.rowkey_dict_path
    opts.inference_actions_len = args.inference_actions_len
    opts.inference_num_targets = args.inference_num_targets
    opts.train_parallel_mode = args.train_parallel_mode
    opts.train_num_parallel = args.train_num_parallel
    opts.optimizer_type = args.optimizer_type
    opts.optimizer_epsilon = args.optimizer_epsilon
    opts.optimizer_adam_beta1 = args.optimizer_adam_beta1
    opts.optimizer_adam_beta2 = args.optimizer_adam_beta2
    opts.optimizer_rmsprop_decay = args.optimizer_rmsprop_decay
    opts.optimizer_rmsprop_momentum = args.optimizer_rmsprop_momentum
    opts.optimizer_rmsprop_centered = bool(args.optimizer_rmsprop_centered)
    opts.optimizer_momentum_momentum = args.optimizer_momentum_momentum
    opts.optimizer_momentum_use_nesterov = \
        bool(args.optimizer_momentum_use_nesterov)
    opts.clip_gradients = bool(args.clip_gradients)
    opts.clip_gradients_norm = args.clip_gradients_norm
    opts.l2_regularizer = args.l2_regularizer
    opts.use_early_stopping = bool(args.use_early_stopping)
    opts.auc_num_thresholds = args.auc_num_thresholds
    opts.optimizer_exponential_decay_steps = \
        args.optimizer_exponential_decay_steps
    opts.optimizer_exponential_decay_rate = \
        args.optimizer_exponential_decay_rate
    opts.optimizer_exponential_decay_staircase = \
        bool(args.optimizer_exponential_decay_staircase)
    opts.optimizer_ftrl_lr_power = args.optimizer_ftrl_lr_power
    opts.optimizer_ftrl_initial_accumulator_value = \
        args.optimizer_ftrl_initial_accumulator_value
    opts.optimizer_ftrl_l1_regularization = \
        args.optimizer_ftrl_l1_regularization
    opts.optimizer_ftrl_l2_regularization = \
        args.optimizer_ftrl_l2_regularization
    opts.optimizer_ftrl_l2_shrinkage_regularization = \
        args.optimizer_ftrl_l2_shrinkage_regularization
    opts.evaluate_every_secs = args.evaluate_every_secs
    opts.leaky_relu_alpha = args.leaky_relu_alpha

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
