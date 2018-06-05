#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from options import Options

import os
import argparse
import tensorflow as tf

import model
import input_data
import hook

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', default='', type=str, help='')
parser.add_argument('--eval_data_path', default='', type=str, help='')
parser.add_argument('--lr', default=0.25, type=float, help='learning rate')
parser.add_argument('--dim', default=100, type=int, help='embedding dim')
parser.add_argument('--maxn', default=0, type=int, help='')
parser.add_argument('--minn', default=0, type=int, help='')
parser.add_argument('--word_ngrams', default=1, type=int, help='')
parser.add_argument('--bucket', default=2000000, type=int, help='')
parser.add_argument('--ws', default=20, type=int, help='window size')
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

parser.add_argument('--nce_params_dir', default='', type=str, help='')

opts = Options()


def parse_args(argv):
    args = parser.parse_args(argv[1:])
    opts.train_data_path = args.train_data_path
    opts.eval_data_path = args.eval_data_path
    opts.lr = args.lr
    opts.dim = args.dim
    opts.maxn = args.maxn
    opts.minn = args.minn
    opts.word_ngrams = args.word_ngrams
    opts.bucket = args.bucket
    opts.ws = args.ws
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

    opts.nce_params_dir = args.nce_params_dir

    tf.logging.info(opts)


def delete_dir(filename):
    tf.logging.info("To delete file '{}' ...".format(filename))
    if tf.gfile.Exists(filename):
        if tf.gfile.IsDirectory(filename):
            tf.logging.info("delete dir '{}' ...".format(filename))
            tf.gfile.DeleteRecursively(filename)
            tf.logging.info("delete dir '{}' OK".format(filename))
        else:
            raise Exception(
                "'{}' exists and not a directory.".format(filename))


def check_args(opts):
    if opts.optimize_level not in model.ALL_OPTIMIZE_LEVELS:
        raise ValueError(
            "optimaize_level {} not surpported.".format(opts.optimize_level))


def main(argv):
    parse_args(argv)
    check_args(opts)

    if opts.remove_model_dir:
        delete_dir(opts.model_dir)
    else:
        tf.logging.info("Don't remove model dir, maybe restore checkpoint ...")

    input_data.init_dict(opts)
    dict_meta = input_data.parse_dict_meta(opts)
    feature_columns = input_data.feature_columns(opts)

    # session_config not used
    session_config = tf.ConfigProto(device_count={"CPU": 1},
                                    inter_op_parallelism_threads=1,
                                    intra_op_parallelism_threads=1,
                                    log_device_placement=False)

    config = tf.estimator.RunConfig(
        model_dir=opts.model_dir,
        tf_random_seed=None,
        save_summary_steps=opts.save_summary_steps,
        save_checkpoints_secs=opts.save_checkpoints_secs,
        session_config=None,
        keep_checkpoint_max=opts.keep_checkpoint_max,
        keep_checkpoint_every_n_hours=10000,
        log_step_count_steps=opts.log_step_count_steps)
    classifier = tf.estimator.Estimator(
        model_fn=model.knet_model,
        config=config,
        params={
            'feature_columns': feature_columns,
            'hidden_units': opts.hidden_units,
            'n_classes': dict_meta["nwords"] + 1,
            'embedding_dim': opts.dim,
            'learning_rate': opts.lr,
            'num_sampled': opts.num_sampled,
            'recall_k': opts.recall_k,
            'dict_dir': opts.dict_dir,
            'optimize_level': opts.optimize_level,
            'nce_params_dir': opts.nce_params_dir
        })

    # Create profile hooks
    save_steps = opts.profile_steps
    meta_hook = hook.MetadataHook(save_steps=save_steps,
                                  output_dir=opts.model_dir)
    profile_hook = tf.train.ProfilerHook(save_steps=save_steps,
                                         output_dir=opts.model_dir,
                                         show_dataflow=True,
                                         show_memory=True)
    hooks = [meta_hook, profile_hook] if opts.use_profile_hook else None

    # train model
    tf.logging.info("Beginning train model ...")
    classifier.train(input_fn=lambda: input_data.train_input_fn(opts),
                     max_steps=opts.max_train_steps,
                     hooks=hooks)
    tf.logging.info("Train model OK")

    # save nce params
    if not os.path.exists(opts.nce_params_dir):
        os.mkdir(opts.nce_params_dir)

    tf.logging.info("Save nce weights and biases ...")
    model.save_model_nce_params(classifier, opts.nce_params_dir)
    tf.logging.info("Save nce weights and biases OK")

    # evaluate model
    tf.logging.info("Beginning evaluate model ...")
    classifier.evaluate(input_fn=lambda: input_data.eval_input_fn(opts),
                        hooks=hooks)
    tf.logging.info("Evaluate model OK")

    # export model
    tf.logging.info("Beginning export model ...")
    assets_dict_dir = os.path.basename(opts.dict_dir)
    assets_nce_params_dir = os.path.basename(opts.nce_params_dir)
    dict_params = {}
    nce_params = {}
    for name in input_data.DICT_PARAM_NAMES:
        src = os.path.join(opts.dict_dir, name)
        dest = os.path.join(assets_dict_dir, name)
        dict_params[dest] = src
    for name in model.NCE_PARAM_NAMES:
        src = os.path.join(opts.nce_params_dir, name)
        dest = os.path.join(assets_nce_params_dir, name)
        nce_params[dest] = src

    assets_extra = dict(dict_params, **nce_params)

    classifier.export_savedmodel(
        opts.export_model_dir,
        serving_input_receiver_fn=input_data.build_serving_input_fn(opts),
        assets_extra=assets_extra)
    tf.logging.info("Export model OK")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
