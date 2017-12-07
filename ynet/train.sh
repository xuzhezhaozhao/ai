#! /usr/bin/env bash

set -e

input_prefix=$1

python train.py --learning_rate 0.1 --epoches 5 --batch_size 50 --keep_prob 0.5 --input_size 256 --num_sampled 10 --watched_size 20 --video_embeddings_file_binary ${input_prefix}.vec --video_embeddings_file_dict ${input_prefix}.dict --train_watched_file ${input_prefix}.watched --train_predicts_file ${input_prefix}.predicts
