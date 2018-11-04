#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.app.flags.DEFINE_string('model_dir', 'model_dir', '')
tf.app.flags.DEFINE_string('export_model_dir', 'export_model_dir', '')

tf.app.flags.DEFINE_string('run_mode', 'train', 'train, predict and all')
tf.app.flags.DEFINE_string('train_data_path', '', 'train data path')
tf.app.flags.DEFINE_string('eval_data_path', '', 'eval data path')
tf.app.flags.DEFINE_string('predict_data_path', '', 'predict data path')
tf.app.flags.DEFINE_string('predict_output', '', '')
tf.app.flags.DEFINE_string('predict_checkpoint_path', '', '')

# train flags
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('max_train_steps', -1, '')
tf.app.flags.DEFINE_integer('epoch', 1, '')
tf.app.flags.DEFINE_integer('throttle_secs', 600, '')

# dataset flags
tf.app.flags.DEFINE_integer('prefetch_size', 1000, '')
tf.app.flags.DEFINE_integer('shuffle_size', 1000, '')
tf.app.flags.DEFINE_bool('shuffle_batch', True, '')
tf.app.flags.DEFINE_integer('map_num_parallel_calls', 1, '')

# log flags
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_integer('save_checkpoints_secs', -1, '')
tf.app.flags.DEFINE_integer('save_checkpoints_steps', 600, '')
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

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

# moving average flags
tf.app.flags.DEFINE_bool(
    'use_moving_average', False,
    'Weather to use moving average.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', 0.9,
    'The decay to use for the moving average.')

# preprocess flags
tf.app.flags.DEFINE_integer('inference_image_size', 256, '')
tf.app.flags.DEFINE_integer('train_image_size', 224, '')
tf.app.flags.DEFINE_integer('resize_side_min', 256, '')
tf.app.flags.DEFINE_integer('resize_side_max', 512, '')

# finetune flags
tf.app.flags.DEFINE_integer('num_classes', 1000, '')
tf.app.flags.DEFINE_string(
    'model_name', '',
    'Specifies the model to finetune. One of "vgg16", "vgg19", "inception_v1"'
    ', "inception_v2", "inception_v3", "inception_v4", "inception_resnet_v2"'
    ', "resnet_v1_50"'
)
tf.app.flags.DEFINE_string('preprocess_name', 'easy', '')
tf.app.flags.DEFINE_string('pretrained_weights_path', '', '')
tf.app.flags.DEFINE_list('trainable_scopes', '', '')
tf.app.flags.DEFINE_list('exclude_restore_scopes', '', '')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, '')
tf.app.flags.DEFINE_float('weight_decay', 0.0001, '')
tf.app.flags.DEFINE_bool('use_batch_norm', True, '')
tf.app.flags.DEFINE_float('batch_norm_decay', 0.9997, '')
tf.app.flags.DEFINE_float('batch_norm_epsilon', 0.001, '')
tf.app.flags.DEFINE_bool('global_pool', False, '')
tf.app.flags.DEFINE_integer('min_depth', 16, '')
tf.app.flags.DEFINE_float('depth_multiplier', 1.0, '')
tf.app.flags.DEFINE_bool('spatial_squeeze', True, '')
tf.app.flags.DEFINE_bool('create_aux_logits', False, '')

opts = tf.app.flags.FLAGS
