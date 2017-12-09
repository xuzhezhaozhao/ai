#! /usr/bin/env bash

input_prefix=$1
argsmode=$2

python train.py --learning_rate 0.0001 --epoches 10 --batch_size 500 --keep_prob 0.5 --embedding_dim 100 --num_sampled 100 --watched_size 20 --video_embeddings_file_binary ${input_prefix}.vec --video_embeddings_file_dict ${input_prefix}.dict --train_watched_file ${input_prefix}.watched --train_predicts_file ${input_prefix}.predicts --model_dir model --run_mode train --k=50 --loss nce --embeddings_trainable true ${argsmode}

echo 'eval top 50\n'

python train.py --learning_rate 0.0001 --epoches 30 --batch_size 500 --keep_prob 0.5 --embedding_dim 100 --num_sampled 100 --watched_size 20 --video_embeddings_file_binary ${input_prefix}.vec --video_embeddings_file_dict ${input_prefix}.dict --train_watched_file ${input_prefix}.watched --train_predicts_file ${input_prefix}.predicts --model_dir model --run_mode eval --k=50 --loss nce --embeddings_trainable true

echo 'eval top 20\n'

python train.py --learning_rate 0.0001 --epoches 30 --batch_size 500 --keep_prob 0.5 --embedding_dim 100 --num_sampled 100 --watched_size 20 --video_embeddings_file_binary ${input_prefix}.vec --video_embeddings_file_dict ${input_prefix}.dict --train_watched_file ${input_prefix}.watched --train_predicts_file ${input_prefix}.predicts --model_dir model --run_mode eval --k=20 --loss nce --embeddings_trainable true
