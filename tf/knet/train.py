#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from options import Options
from model import knet_model

import os
import argparse
import tensorflow as tf
import input_data

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

    tf.logging.info('\n' + str(opts))


def main(argv):
    parse_args(argv)

    input_data.init_dict(opts)
    dict_meta = input_data.parse_dict_meta(opts)
    feature_columns = input_data.feature_columns(opts)

    config = tf.estimator.RunConfig(
        model_dir=opts.model_dir,
        tf_random_seed=None,
        save_summary_steps=opts.save_summary_steps,
        save_checkpoints_secs=opts.save_checkpoints_secs,
        session_config=None,
        keep_checkpoint_max=opts.keep_checkpoint_max,
        keep_checkpoint_every_n_hours=10000,
        log_step_count_steps=opts.log_step_count_steps
    )
    classifier = tf.estimator.Estimator(
        model_fn=knet_model,
        config=config,
        params={
            'feature_columns': feature_columns,
            'hidden_units': opts.hidden_units,
            'n_classes': dict_meta["nwords"] + 1,
            'embedding_dim': opts.dim,
            'learning_rate': opts.lr,
            'num_sampled': opts.num_sampled,
            'recall_k': opts.recall_k,
            'dict_dir': opts.dict_dir
        })

    # train model
    classifier.train(input_fn=lambda: input_data.train_input_fn(opts),
                     max_steps=opts.max_train_steps)

    # evaluate model
    classifier.evaluate(input_fn=lambda: input_data.eval_input_fn(opts))

    # export model
    dict_dir = opts.dict_dir
    dict_words_path = os.path.join(dict_dir, input_data.DICT_WORDS)
    dict_meta_path = os.path.join(dict_dir, input_data.DICT_META)
    saved_dict_bin_path = os.path.join(dict_dir, input_data.SAVED_DICT_BIN)
    assets_extra = {dict_words_path: dict_words_path,
                    dict_meta_path: dict_meta_path,
                    saved_dict_bin_path: saved_dict_bin_path}
    classifier.export_savedmodel(
        opts.export_model_dir,
        serving_input_receiver_fn=input_data.build_serving_input_fn(opts),
        assets_extra=assets_extra
    )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
