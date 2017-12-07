#! /usr/bin/env bash

set -e

python train.py --learning_rate 0.1 --epoches 5 --batch_size 50 --keep_prob 0.5 --input_size 256 --num_sampled 10 --watched_size 20 --video_embeddings_file_binary data/data.in.tf.vec --video_embeddings_file_dict data/data.in.tf.dict --train_watched_file data/data.in.tf.watched --train_predicts_file data/data.in.tf.predicts
