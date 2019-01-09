#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np

import build_model_fn
import input_data

tf.app.flags.DEFINE_string('model_dir', 'model_dir', '')
tf.app.flags.DEFINE_string('export_model_dir', 'export_model_dir', '')

tf.app.flags.DEFINE_string('run_mode', 'train', 'train, predict and all')
tf.app.flags.DEFINE_string('train_data_path', '', 'train data path')
tf.app.flags.DEFINE_string('eval_data_path', '', 'eval data path')
tf.app.flags.DEFINE_string('predict_data_path', '', 'predict data path')
tf.app.flags.DEFINE_string('predict_output', '', '')
tf.app.flags.DEFINE_string('predict_checkpoint_path', '', '')

tf.app.flags.DEFINE_string('char_dict_path', '', '')
tf.app.flags.DEFINE_string('label_dict_path', '', '')

# train flags
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('eval_batch_size', 256, 'eval batch size')
tf.app.flags.DEFINE_integer('max_train_steps', -1, '')
tf.app.flags.DEFINE_integer('epoch', 1, '')
tf.app.flags.DEFINE_integer('throttle_secs', 600, '')

# dataset flags
tf.app.flags.DEFINE_string('label_str', '__label__', '')
tf.app.flags.DEFINE_integer('max_length', 32, '')
tf.app.flags.DEFINE_integer('num_filters', 128,
                            'Number of filters per filter size (default: 128)')
tf.app.flags.DEFINE_list('filter_sizes', '3,4,5',
                         'Comma-separated filter sizes (default: 3,4,5)')
tf.app.flags.DEFINE_list('label_weights', '1.0,0.2', '')
tf.app.flags.DEFINE_integer('embedding_dim', 50, '')
tf.app.flags.DEFINE_integer('prefetch_size', 1000, '')
tf.app.flags.DEFINE_integer('shuffle_size', 1000, '')
tf.app.flags.DEFINE_bool('shuffle_batch', True, '')
tf.app.flags.DEFINE_integer('map_num_parallel_calls', 1, '')

# log flags
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_integer('save_checkpoints_secs', -1, '')
tf.app.flags.DEFINE_integer('save_checkpoints_steps', -1, '')
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

tf.app.flags.DEFINE_integer('decay_steps', 100, '')

# moving average flags
tf.app.flags.DEFINE_bool(
    'use_moving_average', False,
    'Weather to use moving average.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', 0.9,
    'The decay to use for the moving average.')

tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, '')
tf.app.flags.DEFINE_float('weight_decay', 0.0001, '')
tf.app.flags.DEFINE_bool('use_batch_norm', True, '')
tf.app.flags.DEFINE_float('batch_norm_decay', 0.9997, '')
tf.app.flags.DEFINE_float('batch_norm_epsilon', 0.001, '')

# early stopping flags \
tf.app.flags.DEFINE_float('max_epoches_without_decrease', 2.0, '')
tf.app.flags.DEFINE_float('early_stopping_min_epoches', 1.0, '')
tf.app.flags.DEFINE_float('early_stopping_run_every_epoches', 1.0, '')

opts = tf.app.flags.FLAGS

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'


def build_estimator(opts):
    """Build estimator."""

    save_checkpoints_secs = None
    if opts.save_checkpoints_secs > 0:
        save_checkpoints_secs = opts.save_checkpoints_secs

    save_checkpoints_steps = None
    if opts.save_checkpoints_steps > 0 and opts.save_checkpoints_epoches > 0:
        raise ValueError("save_checkpoints_steps and save_checkpoints_epoches "
                         "should not be both set.")

    if opts.save_checkpoints_steps > 0:
        save_checkpoints_steps = opts.save_checkpoints_steps

    config_keys = {}
    config_keys['model_dir'] = opts.model_dir
    config_keys['tf_random_seed'] = None
    config_keys['save_summary_steps'] = opts.save_summary_steps
    config_keys['save_checkpoints_secs'] = save_checkpoints_secs
    config_keys['save_checkpoints_steps'] = save_checkpoints_steps
    config_keys['session_config'] = None
    config_keys['keep_checkpoint_max'] = opts.keep_checkpoint_max
    config_keys['keep_checkpoint_every_n_hours'] = 10000
    config_keys['log_step_count_steps'] = opts.log_step_count_steps

    labels = [label.strip() for label in open(opts.label_dict_path)
              if label.strip() != '']
    estimator_keys = {}
    estimator_keys['model_fn'] = build_model_fn.model_fn
    estimator_keys['params'] = {
        'opts': opts,
        'num_classes': len(labels)
    }
    config = tf.estimator.RunConfig(**config_keys)
    estimator_keys['config'] = config

    estimator = tf.estimator.Estimator(**estimator_keys)
    return estimator


def train(opts, estimator):
    """Train and eval model in lcoal mode."""

    train_input_fn = input_data.build_train_input_fn(
        opts, opts.train_data_path)
    max_steps = None
    if opts.max_train_steps > 0:
        max_steps = opts.max_train_steps
    tf.logging.info('max_steps = {}'.format(max_steps))

    tf.logging.info("Training model ...")
    estimator.train(
        input_fn=train_input_fn,
        max_steps=max_steps)
    tf.logging.info("Train model done\n")


def evaluate(opts, estimator):
    tf.logging.info("Evaluating model in test dataset ...")
    eval_input_fn = input_data.build_eval_input_fn(
        opts, opts.eval_data_path)
    eval_result = estimator.evaluate(input_fn=eval_input_fn)
    tf.logging.info("Evaluating model in test dataset done")

    return eval_result


def export(opts, estimator):
    """Export model in local mode."""

    # export model
    tf.logging.info("Beginning export model ...")
    estimator.export_savedmodel(
        opts.export_model_dir,
        serving_input_receiver_fn=input_data.build_serving_input_fn(opts))
    tf.logging.info("Export model OK")


def predict(opts, estimator):
    tf.logging.info("Begin predict ...")
    build_predict_input_fn = input_data.build_predict_input_fn(
        opts, opts.predict_data_path)

    checkpoint_path = opts.predict_checkpoint_path
    if tf.gfile.IsDirectory(opts.predict_checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    results = estimator.predict(
        input_fn=build_predict_input_fn,
        checkpoint_path=checkpoint_path,
        yield_single_examples=True)

    labels = [label.strip() for label in open(opts.label_dict_path)
              if label.strip() != '']
    with open(opts.predict_output, 'w') as fout, \
            open(opts.predict_data_path, 'r') as fin:
        for result in results:
            argmax = np.argmax(result['scores'])
            fout.write(labels[argmax])
            fout.write(' ')
            fout.write(str(result['scores'][argmax]))
            fout.write('\n')
    tf.logging.info("Predict done")


def main(_):
    D = opts.flag_values_dict()
    tf.logging.info("FLAGS: ")
    for key in D:
        tf.logging.info('{} = {}'.format(key, D[key]))

    estimator = build_estimator(opts)
    if opts.run_mode == 'train':
        train(opts, estimator)
        evaluate(opts, estimator)
    elif opts.run_mode == 'eval':
        evaluate(opts, estimator)
    elif opts.run_mode == 'predict':
        predict(opts, estimator)
    elif opts.run_mode == 'all':
        train(opts, estimator)
        evaluate(opts, estimator)
        predict(opts, estimator)
    else:
        raise ValueError("Unsupported run mode.")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
