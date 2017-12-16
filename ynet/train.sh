#! /usr/bin/env bash

input_prefix=$1

python train.py --learning_rate 0.01 --epoches 10 --batch_size 500 --keep_prob 0.5 --embedding_dim 100 --num_sampled 100 --watched_size 30 --video_embeddings_file_binary ${input_prefix}.vec --video_embeddings_file_dict ${input_prefix}.dict --train_watched_file ${input_prefix}.watched --train_predicts_file ${input_prefix}.predicts --model_dir model --run_mode train --k=50 --loss nce --embeddings_trainable true
