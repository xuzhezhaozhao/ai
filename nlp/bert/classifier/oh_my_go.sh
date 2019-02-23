#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

MODEL_DIR=`pwd`/model_dir

DATA_DIR=../../../datasets/thucnews-dataset/data/tmp
BERT_BASE_DIR=../pretrained_checkpoints/chinese_L-12_H-768_A-12

python run_classifier.py \
  --task_name=fasttext \
  --do_train=true \
  --do_eval=true \
  --data_dir=${DATA_DIR} \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=${MODEL_DIR} \
  --save_checkpoints_steps 1000 \
  --save_summary_steps 10 \
  --keep_checkpoint_max 3 \
  --log_step_count_steps 1
