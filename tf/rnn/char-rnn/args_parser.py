#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


tf.app.flags.DEFINE_string('model_dir', 'model_dir', '')
tf.app.flags.DEFINE_string('export_model_dir', 'export_model_dir', '')
tf.app.flags.DEFINE_string(
    'preprocessed_filename', 'model_dir/preprocessed.pkl', '')

tf.app.flags.DEFINE_string('run_mode', 'train', 'train, predict and all')
tf.app.flags.DEFINE_string('train_data_path', '', 'train data path')
tf.app.flags.DEFINE_string('start_string', '', 'char rnn sample start string')
tf.app.flags.DEFINE_integer('num_samples', 100, '')

tf.app.flags.DEFINE_float(
    'sample_temperature', 1.0,
    'Low temperatures results in more predictable text.'
    'Higher temperatures results in more surprising text.'
    'Experiment to find the best setting.')

# train flags
tf.app.flags.DEFINE_integer('seq_length', 20, 'sequence length')
tf.app.flags.DEFINE_integer('hidden_size', 100, 'rnn hidden size')
tf.app.flags.DEFINE_integer('num_layers', 2, 'rnn num layers')
tf.app.flags.DEFINE_float('keep_prob', 1.0, 'rnn dropout keep prob')
tf.app.flags.DEFINE_bool('use_clip_gradients', True, '')
tf.app.flags.DEFINE_float('clip_norm', 5.0, '')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('epoch', 1, '')

tf.app.flags.DEFINE_bool(
    'use_embedding', False,
    'whether use embedding for chars')

tf.app.flags.DEFINE_integer(
    'embedding_dim', 100,
    'embedding dim for chars')

# dataset flags
tf.app.flags.DEFINE_integer('prefetch_size', 1000, '')
tf.app.flags.DEFINE_integer('shuffle_size', 1000, '')
tf.app.flags.DEFINE_bool('shuffle_batch', True, '')
tf.app.flags.DEFINE_integer('map_num_parallel_calls', 1, '')


# log flags
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_integer('save_checkpoints_steps', 5000, '')
tf.app.flags.DEFINE_integer('keep_checkpoint_max', 3, '')
tf.app.flags.DEFINE_integer('log_step_count_steps', 100, '')


# profile flags
tf.app.flags.DEFINE_bool('use_profile_hook', False, '')
tf.app.flags.DEFINE_integer('profile_steps', 100, '')

# optimizer flags
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float(
    'opt_epsilon', 1.0,
    'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# learning rate flags
tf.app.flags.DEFINE_float('learning_rate', 0.025, 'learning rate')

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", '
    '"exponential", or "polynomial"')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays. Note: this flag counts'
    ' epochs per clone but aggregates per sync replicas. So 1.0 means that '
    'each clone will go over full epoch individually, but replicas will go '
    'once across all replicas.')


opts = tf.app.flags.FLAGS
