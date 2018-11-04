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
tf.app.flags.DEFINE_float('lr', 0.025, 'learning rate')
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
tf.app.flags.DEFINE_float('optimizer_momentum_momentum', 0.9, '')
tf.app.flags.DEFINE_bool('optimizer_momentum_use_nesterov', False, '')

# preprocess flags
tf.app.flags.DEFINE_integer('inference_image_size', 256, '')
tf.app.flags.DEFINE_integer('train_image_size', 224, '')
tf.app.flags.DEFINE_integer('resize_side_min', 256, '')
tf.app.flags.DEFINE_integer('resize_side_max', 512, '')

# finetune flags
tf.app.flags.DEFINE_integer('num_classes', 1000, '')
tf.app.flags.DEFINE_string('model_name', '', '')
tf.app.flags.DEFINE_string('preprocess_name', 'easy', '')
tf.app.flags.DEFINE_string('pretrained_weights_path', '', '')
tf.app.flags.DEFINE_list('train_layers', '', '')
tf.app.flags.DEFINE_list('exclude_restore_layers', '', '')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, '')
tf.app.flags.DEFINE_float('weights_decay', 0.0001, '')
tf.app.flags.DEFINE_bool('use_batch_norm', True, '')
tf.app.flags.DEFINE_float('batch_norm_decay', 0.9997, '')
tf.app.flags.DEFINE_float('batch_norm_epsilon', 0.001, '')
tf.app.flags.DEFINE_bool('global_pool', False, '')
tf.app.flags.DEFINE_integer('min_depth', 16, '')
tf.app.flags.DEFINE_float('depth_multiplier', 1.0, '')
tf.app.flags.DEFINE_bool('spatial_squeeze', True, '')
tf.app.flags.DEFINE_bool('create_aux_logits', False, '')

opts = tf.app.flags.FLAGS
