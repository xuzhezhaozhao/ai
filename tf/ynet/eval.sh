#! /usr/bin/env bash

input_prefix=data/data.in.tf

python train.py --learning_rate 0.01 --epoches 1 --batch_size 1000 --keep_prob 0.5 --embedding_dim 100 --num_sampled 10 --watched_size 5 --video_embeddings_file_binary ${input_prefix}.vec --video_embeddings_file_dict ${input_prefix}.dict --train_watched_file ${input_prefix}.watched --train_predicts_file ${input_prefix}.predicts --model_dir model --run_mode eval --k=50 --loss nce --embeddings_trainable false
