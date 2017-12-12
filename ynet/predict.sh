#! /usr/bin/env bash

input_prefix=$1

python train.py --learning_rate 0.0001 --epoches 1 --batch_size 50 --keep_prob 0.5 --embedding_dim 50 --num_sampled 100 --watched_size 10 --video_embeddings_file_binary ${input_prefix}.vec --video_embeddings_file_dict ${input_prefix}.dict --train_watched_file ${input_prefix}.watched --train_predicts_file ${input_prefix}.predicts --model_dir model --run_mode predict --k=50 --loss nce --embeddings_trainable true