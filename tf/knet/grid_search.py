#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

params_lr = [x * 0.005 for x in range(1, 100)]
params_embedding_dim = [128, 256]
params_train_ws = [10, 20, 50]
params_batch_size = [16, 32, 64, 128]
params_num_sampled = [10, 50, 100, 200]
params_hidden_units = ["", "-1", "256, -1", "512, 256, -1"]
params_shuffle_batch = [0, 1]
params_optimizer_type = ['ada', 'sgd']
params_use_batch_normalization = [0, 1]

all_params = [
    # params_lr,
    # params_embedding_dim,
    # params_train_ws,
    # params_batch_size,
    # params_num_sampled,
    # params_hidden_units,
    # params_shuffle_batch,
    params_optimizer_type,
    params_use_batch_normalization,
]


_id = 0


def traverse_config(params, config):
    global _id
    if len(params) == 0:
        print(_id, config)
        _id += 1
        return
    for param in params[0]:
        config.append(param)
        traverse_config(params[1:], config)
        config.pop(-1)


def main():
    config = []
    traverse_config(all_params, config)


if __name__ == '__main__':
    main()
