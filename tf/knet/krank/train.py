#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

import build_model_fn
import input_data
import model_keys
import easy_estimator


def build_estimator(opts):
    """Build estimator."""

    config_keys = {}
    config_keys['model_dir'] = opts.model_dir
    config_keys['tf_random_seed'] = None
    config_keys['save_summary_steps'] = opts.save_summary_steps
    config_keys['save_checkpoints_steps'] = opts.save_checkpoints_steps
    config_keys['session_config'] = None
    config_keys['keep_checkpoint_max'] = opts.keep_checkpoint_max
    config_keys['log_step_count_steps'] = opts.log_step_count_steps
    config = tf.estimator.RunConfig(**config_keys)

    estimator_keys = {}
    estimator_keys['model_dir'] = opts.model_dir
    estimator_keys['model_fn'] = build_model_fn.krank_model_fn
    (positive_records_col,
     negative_records_col,
     targets_col) = input_data.input_feature_columns(opts)
    estimator_keys['params'] = {
        model_keys.POSITIVE_RECORDS_COL: positive_records_col,
        model_keys.NEGATIVE_RECORDS_COL: negative_records_col,
        model_keys.TARGETS_COL: targets_col,
        'opts': opts
    }
    estimator_keys['config'] = config
    estimator = easy_estimator.EasyEstimator(**estimator_keys)

    return estimator


def train_and_eval_in_local_mode(opts):
    """Train and eval model in lcoal mode."""

    tf.logging.info("Training model ...")
    build_model_fn.clear_model_fn_times()

    opts.estimator.easy_train(input_fn=input_data.input_fn(opts, False, 1),
                              max_steps=opts.max_train_steps)
    tf.logging.info("Train model OK")

    # evaluate model
    tf.logging.info("Evaluating model ...")
    result = opts.estimator.evaluate(
        input_fn=input_data.input_fn(opts, True),
        steps=opts.max_eval_steps)
    tf.logging.info("Evaluate model OK")

    return result


def export_model_in_local_mode(opts):
    """Export model in local mode."""

    # export model
    tf.logging.info("Beginning export model ...")
    opts.estimator.export_savedmodel(
        opts.export_model_dir,
        serving_input_receiver_fn=input_data.build_serving_input_fn(opts))
    tf.logging.info("Export model OK")


def train(opts, export=True):
    """Train model."""

    opts.estimator = build_estimator(opts)
    opts.list_all_member()

    result = train_and_eval_in_local_mode(opts)
    if export:
        export_model_in_local_mode(opts)
    return result
