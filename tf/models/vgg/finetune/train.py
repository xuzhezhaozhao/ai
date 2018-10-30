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

    if opts.use_easy_preprocess:
        build_train_input_fn = input_data.build_easy_train_input_fn(opts)
        build_eval_input_fn = input_data.build_easy_eval_input_fn(opts)
    else:
        build_train_input_fn = input_data.build_train_input_fn(opts)
        build_eval_input_fn = input_data.build_eval_input_fn(opts)

    best_accuracy = 0.0
    accuracy_no_increase = 0
    global_lr = opts.lr
    build_model_fn.set_global_learning_rate(global_lr)
    for epoch in range(opts.epoch):
        epoch += 1
        tf.logging.info("Beginning train model, epoch {} ...".format(epoch))
        opts.estimator.train(input_fn=build_train_input_fn,
                             max_steps=opts.max_train_steps,
                             hooks=opts.hooks)
        tf.logging.info("Train model OK, epoch {}".format(epoch))

        tf.logging.info("Beginning evaluate model, epoch {} ...".format(epoch))
        result = opts.estimator.evaluate(
            input_fn=build_eval_input_fn,
            hooks=opts.hooks)
        tf.logging.info("Evaluate model OK, epoch {}".format(epoch))

        if result['accuracy'] > best_accuracy + opts.min_accuracy_increase:
            accuracy_no_increase = 0
            best_accuracy = result['accuracy']
        else:
            accuracy_no_increase += 1
            if accuracy_no_increase == opts.lr_decay_epoch_when_no_increase:
                global_lr *= opts.lr_decay_rate
                build_model_fn.set_global_learning_rate(global_lr)
                tf.logging.info(
                    "Accuracy no increase, learning rate decay by {}."
                    .format(opts.lr_decay_rate))
            elif accuracy_no_increase > opts.lr_decay_epoch_when_no_increase:
                tf.logging.info("Accuracy no increase, early stopping.")
                break
            else:
                tf.logging.info("Accuracy no increase, try once more.")

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
