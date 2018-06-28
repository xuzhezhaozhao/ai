#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import copy
import time

from options import Options
import train
import model_keys

_id = 0

params_lr = [x * 0.005 for x in range(1, 100)]
params_embedding_dim = [128, 256]
params_train_ws = [10, 20, 50]
params_batch_size = [16, 32, 64, 128]
params_num_sampled = [10, 50, 100, 200]
params_hidden_units = [[], [-1], [256, -1], [512, 256, -1]]
params_shuffle_batch = [0, 1]
params_optimizer_type = ['ada', 'sgd']
params_use_batch_normalization = [0, 1]
params_train_nce_biases = [0, 1]

all_params = [
    params_lr,
    params_embedding_dim,
    params_train_ws,
    params_batch_size,
    params_num_sampled,
    params_hidden_units,
    params_shuffle_batch,
    params_optimizer_type,
    params_use_batch_normalization,
    params_train_nce_biases,
]


def get_lr(config): return config[0]


def get_embedding_dim(config): return config[1]


def get_train_ws(config): return config[2]


def get_batch_size(config): return config[3]


def get_num_sampled(config): return config[4]


def get_hidden_units(config):
    if len(config[5]) > 0 and config[5][-1] == -1:
        new_config = copy.deepcopy(config[5])
        new_config[-1] = get_embedding_dim(config)
        return new_config
    else:
        return config[5]


def get_shuffle_batch(config): return config[6]


def get_optimizer_type(config): return config[7]


def get_use_batch_normalization(config): return config[8]


def get_train_nce_biases(config): return config[9]


def eval_config(id, config, f):
    opts = Options()
    opts.train_data_path = '../../data/train_data.in'
    opts.eval_data_path = '../../data/eval_data.in'
    opts.lr = get_lr(config)
    opts.embedding_dim = get_embedding_dim(config)
    opts.train_ws = get_train_ws(config)
    opts.min_count = 30
    opts.t = 0.025
    opts.verbose = 1
    opts.min_count_label = 5
    opts.label = "__label__"
    opts.batch_size = get_batch_size(config)
    opts.num_sampled = get_num_sampled(config)
    opts.max_train_steps = None
    opts.epoch = 5
    opts.hidden_units = get_hidden_units(config)
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
    opts.train_nce_biases = get_train_nce_biases(config)
    opts.shuffle_batch = get_shuffle_batch(config)
    opts.predict_ws = 20
    opts.sample_dropout = 0.0
    opts.optimizer_type = get_optimizer_type(config)
    opts.tfrecord_file = ''
    opts.num_tfrecord_file = 1
    opts.train_data_format = 'fasttext'
    opts.tfrecord_map_num_parallel_calls = 1
    opts.train_parallel_mode = 'train_op_parallel'
    opts.num_train_op_parallel = 4
    opts.use_batch_normalization = get_use_batch_normalization(config)

    opts.tf_config = None
    opts.task_type = model_keys.TaskType.LOCAL  # default mode

    start = int(time.time())
    result = train.train(opts, export=False)
    end = int(time.time())
    elapsed = end - start
    f.write(str(opts.lr))
    f.write('\t')
    f.write(str(opts.embedding_dim))
    f.write('\t')
    f.write(str(opts.train_ws))
    f.write('\t')
    f.write(str(opts.batch_size))
    f.write('\t')
    f.write(str(opts.num_sampled))
    f.write('\t')
    f.write(str(opts.hidden_units))
    f.write('\t')
    f.write(str(opts.shuffle_batch))
    f.write('\t')
    f.write(str(opts.optimizer_type))
    f.write('\t')
    f.write(str(opts.use_batch_normalization))
    f.write('\t')

    f.write(str(result['loss']))
    f.write('\t')
    f.write(str(result['precision_at_top_10']))
    f.write('\t')
    f.write(str(result['recall_at_top_10']))
    f.write('\t')
    f.write(str(result['average_precision_at_10']))
    f.write('\t')
    f.write(str(result['accuracy']))
    f.write('\t')
    f.write(str(elapsed))
    f.write('\n')
    f.flush()


def traverse_config(params, config, f):
    global _id
    if len(params) == 0:
        print("##### eval config: ", config)
        eval_config(_id, config, f)
        _id += 1
        return
    for param in params[0]:
        config.append(param)
        traverse_config(params[1:], config, f)
        config.pop(-1)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    config = []
    with open('result.log', 'w') as f:
        f.write("lr\tembedding_dim\ttrain_ws\tbatch_size\tnum_sampled\t"
                "hidden_units\tshuffle_batch\toptimizer_type\t"
                "use_batch_normalization\tloss\tprecision_at_top_10\t"
                "recall_at_top_10\taverage_precision_at_top_10\t"
                "accuracy\ttime\n")
        traverse_config(all_params, config, f)


if __name__ == '__main__':
    main()
