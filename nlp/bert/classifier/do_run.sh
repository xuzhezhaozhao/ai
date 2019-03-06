#! /usr/bin/env bash

BERT_CLASSIFIER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source /usr/local/services/kd_anaconda2-1.0/anaconda2_profile >/dev/null 2>&1
source /usr/local/services/kd_anaconda2_gpu-1.0/anaconda2_profile >/dev/null 2>&1

BERT_BASE_DIR=${BERT_CLASSIFIER_DIR}/../pretrained_checkpoints/chinese_L-12_H-768_A-12

python ${BERT_CLASSIFIER_DIR}/run_classifier.py $@ \
    --task_name=fasttext \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt
