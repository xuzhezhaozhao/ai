#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import os
import time

import build_model_fn
import hook
import input_data
import model_keys


def build_estimator(opts):
    """Build estimator."""

    """ session config
    session_config = tf.ConfigProto(device_count={"CPU": 1},
                                    inter_op_parallelism_threads=1,
                                    intra_op_parallelism_threads=1,
                                    log_device_placement=False)
    """
    config_keys = {}
    config_keys['model_dir'] = opts.model_dir
    config_keys['tf_random_seed'] = None
    config_keys['save_summary_steps'] = opts.save_summary_steps
    config_keys['save_checkpoints_secs'] = opts.save_checkpoints_secs
    config_keys['session_config'] = None
    config_keys['keep_checkpoint_max'] = opts.keep_checkpoint_max
    config_keys['keep_checkpoint_every_n_hours'] = 10000
    config_keys['log_step_count_steps'] = opts.log_step_count_steps

    estimator_keys = {}
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
    config = tf.estimator.RunConfig(**config_keys)
    estimator_keys['config'] = config

    estimator = tf.estimator.Estimator(**estimator_keys)

    return estimator


def create_hooks(opts):
    """Create profile hooks."""

    save_steps = opts.profile_steps
    meta_hook = hook.MetadataHook(save_steps=save_steps,
                                  output_dir=opts.model_dir)
    profile_hook = tf.train.ProfilerHook(save_steps=save_steps,
                                         output_dir=opts.model_dir,
                                         show_dataflow=True,
                                         show_memory=True)
    hooks = [meta_hook, profile_hook] if opts.use_profile_hook else None

    return hooks


def train_and_eval_in_local_mode(opts):
    """Train and eval model in lcoal mode."""

    tf.logging.info("Beginning train model ...")
    opts.estimator.train(input_fn=lambda: input_data.input_fn(opts, False),
                         max_steps=opts.max_train_steps,
                         hooks=opts.hooks)
    tf.logging.info("Train model OK")

    # evaluate model
    tf.logging.info("Beginning evaluate model ...")
    result = opts.estimator.evaluate(
        input_fn=lambda: input_data.input_fn(opts, True),
        hooks=opts.hooks)
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
    opts.hooks = create_hooks(opts)

    tf.logging.info(opts)

    result = train_and_eval_in_local_mode(opts)
    if export:
        # export_model_in_local_mode(opts)
        pass
    return result
