#! /usr/bin/env bash

set -e

input_prefix=$1
argsmode=$2

python train.py --learning_rate 0.00001 --epoches 100 --batch_size 50 --keep_prob 0.5 --embedding_dim 100 --num_sampled 3 --watched_size 20 --video_embeddings_file_binary ${input_prefix}.vec --video_embeddings_file_dict ${input_prefix}.dict --train_watched_file ${input_prefix}.watched --train_predicts_file ${input_prefix}.predicts --model_dir model --run_mode train --k=200 --loss nce --embeddings_trainable true ${argsmode}
