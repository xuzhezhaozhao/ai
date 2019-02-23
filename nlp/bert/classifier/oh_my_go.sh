#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

MODEL_DIR=`pwd`/model_dir
TF_RECORD_DIR=`pwd`/tf_record_dir

DATA_DIR=../../../datasets/thucnews-dataset/data/tmp
BERT_BASE_DIR=../pretrained_checkpoints/chinese_L-12_H-768_A-12

python run_classifier.py \
  --data_dir=${DATA_DIR} \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --task_name=fasttext \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --output_dir=${MODEL_DIR} \
  --output_tf_record_dir=${TF_RECORD_DIR} \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_lower_case=true \
  --max_seq_length=128 \
  --do_train=true \
  --do_eval=true \
  --do_predict=false \
  --train_batch_size=4 \
  --eval_batch_size=4 \
  --predict_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --warmup_proportion 0.1 \
  --save_checkpoints_steps 1000 \
  --save_summary_steps 10 \
  --keep_checkpoint_max 3 \
  --log_step_count_steps 1
