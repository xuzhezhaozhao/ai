#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

import build_model_fn
import hook
import input_data


def build_estimator(opts):
    """Build estimator."""

    config_keys = {}
    config_keys['model_dir'] = opts.model_dir
    config_keys['tf_random_seed'] = None
    config_keys['save_summary_steps'] = opts.save_summary_steps
    # config_keys['save_checkpoints_secs'] = opts.save_checkpoints_secs
    config_keys['save_checkpoints_steps'] = opts.save_checkpoints_steps
    config_keys['session_config'] = None
    config_keys['keep_checkpoint_max'] = opts.keep_checkpoint_max
    config_keys['keep_checkpoint_every_n_hours'] = 10000
    config_keys['log_step_count_steps'] = opts.log_step_count_steps

    estimator_keys = {}
    estimator_keys['model_fn'] = build_model_fn.vgg19_model_fn
    estimator_keys['params'] = {'opts': opts}
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

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_data.train_input_fn(opts),
        max_steps=opts.max_train_steps,
        hooks=opts.hooks)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_data.eval_input_fn(opts),
        steps=None,
        name='test',
        start_delay_secs=3,
        throttle_secs=600
    )
    result = tf.estimator.train_and_evaluate(
        opts.estimator, train_spec, eval_spec)
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
    opts.list_all_member()

    result = train_and_eval_in_local_mode(opts)
    if export:
        export_model_in_local_mode(opts)
    return result
