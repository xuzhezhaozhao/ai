#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import hyperopt

from options import Options
import train
import model_keys

_id = 0

params_embedding_dim = [128, 256]
params_train_ws = [10, 20, 50]
params_batch_size = [32, 64, 128]
params_num_sampled = [10, 100]
params_hidden_units = [[256, -1], [512, 256, -1]]
params_shuffle_batch = [0, 1]
params_optimizer_type = ['sgd']
params_use_batch_normalization = [0, 1]
params_train_nce_biases = [0, 1]


def get_hidden_units(hidden_units, embedding_dim):
    hidden_units = list(hidden_units)
    if len(hidden_units) > 0 and hidden_units[-1] == -1:
        hidden_units[-1] = embedding_dim
    return hidden_units


def eval_config(config):
    lr = config['lr']
    embedding_dim = config['embedding_dim']
    train_ws = config['train_ws']
    batch_size = config['batch_size']
    num_sampled = config['num_sampled']
    hidden_units = config['hidden_units']
    shuffle_batch = config['shuffle_batch']
    optimizer_type = config['optimizer_type']
    use_batch_normalization = config['use_batch_normalization']
    train_nce_biases = config['train_nce_biases']

    hidden_units = get_hidden_units(hidden_units, embedding_dim)

    opts = Options()
    opts.train_data_path = '../../data/train_data.in'
    opts.eval_data_path = '../../data/eval_data.in'
    opts.lr = lr
    opts.embedding_dim = embedding_dim
    opts.train_ws = train_ws
    opts.min_count = 30
    opts.t = 0.025
    opts.verbose = 1
    opts.min_count_label = 5
    opts.label = "__label__"
    opts.batch_size = batch_size
    opts.num_sampled = num_sampled
    opts.max_train_steps = None
    opts.epoch = 2
    opts.hidden_units = hidden_units
    opts.model_dir = 'model_dir'
    opts.export_model_dir = 'export_model_dir'
    opts.prefetch_size = 100000
    opts.save_summary_steps = 100
    opts.save_checkpoints_secs = 600
    opts.keep_checkpoint_max = 2
    opts.log_step_count_steps = 1000
    opts.recall_k = 10
    opts.dict_dir = 'dict_dir'
    opts.use_saved_dict = False
    opts.use_profile_hook = False
    opts.profile_steps = 100
    opts.root_ops_path = 'lib/'
    opts.remove_model_dir = 1
    opts.optimize_level = 1
    opts.receive_ws = 100
    opts.use_subset = True
    opts.dropout = 0.0
    opts.ntargets = 1
    opts.chief_lock = 'model_dir/chief.lock'
    opts.max_distribute_train_steps = -1
    opts.train_nce_biases = train_nce_biases
    opts.shuffle_batch = shuffle_batch
    opts.predict_ws = 20
    opts.sample_dropout = 0.0
    opts.optimizer_type = optimizer_type
    opts.tfrecord_file = ''
    opts.num_tfrecord_file = 1
    opts.train_data_format = 'fasttext'
    opts.tfrecord_map_num_parallel_calls = 1
    opts.train_parallel_mode = 'train_op_parallel'
    opts.num_train_op_parallel = 4
    opts.use_batch_normalization = use_batch_normalization

    opts.tf_config = None
    opts.task_type = model_keys.TaskType.LOCAL  # default mode

    result = {}
    try:
        result = train.train(opts, export=False)
    except Exception as e:
        print(e)
    return -result.get('loss', -1)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    best = hyperopt.fmin(
        fn=eval_config,
        space={
            'lr': hyperopt.hp.uniform('lr', 0, 1),
            'embedding_dim': hyperopt.hp.choice(
                'embedding_dim', params_embedding_dim),
            'train_ws': hyperopt.hp.choice('train_ws', params_train_ws),
            'batch_size': hyperopt.hp.choice('batch_size', params_batch_size),
            'num_sampled': hyperopt.hp.choice(
                'num_sampled', params_num_sampled),
            'hidden_units': hyperopt.hp.choice(
                'hidden_units', params_hidden_units),
            'shuffle_batch': hyperopt.hp.choice(
                'shuffle_batch', params_shuffle_batch),
            'optimizer_type': hyperopt.hp.choice(
                'optimizer_type', params_optimizer_type),
            'use_batch_normalization': hyperopt.hp.choice(
                'use_batch_normalization', params_use_batch_normalization),
            'train_nce_biases': hyperopt.hp.choice(
                'train_nce_biases', params_train_nce_biases)
        },
        algo=hyperopt.tpe.suggest,
        max_evals=4
    )

    best['embedding_dim'] = params_embedding_dim[best['embedding_dim']]
    best['train_ws'] = params_train_ws[best['train_ws']]
    best['batch_size'] = params_batch_size[best['batch_size']]
    best['num_sampled'] = params_num_sampled[best['num_sampled']]
    best['hidden_units'] = params_hidden_units[best['hidden_units']]
    best['shuffle_batch'] = params_shuffle_batch[best['shuffle_batch']]
    best['optimizer_type'] = params_optimizer_type[best['optimizer_type']]
    best['use_batch_normalization'] = params_use_batch_normalization[
        best['use_batch_normalization']]
    best['train_nce_biases'] = params_train_nce_biases[
        best['train_nce_biases']]

    print(best)


if __name__ == '__main__':
    main()
