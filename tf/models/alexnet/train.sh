#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

MODEL_DIR=`pwd`/model_dir
EXPORT_MODEL_DIR=`pwd`/export_model_dir

train_data_path=./train.txt
eval_data_path=./eval.txt
lr=0.1
batch_size=64
epoch=1
prefetch_size=1000
shuffle_size=1000
max_train_steps=-1
save_summary_steps=100
save_checkpoints_secs=50
log_step_count_steps=100
use_profile_hook=0
profile_steps=1000
remove_model_dir=1
dropout=0.0
shuffle_batch=1
optimizer_type='ada'  # 'ada', 'sgd', 'adadelta', 'adam', 'rmsprop'
map_num_parallel_calls=1
# 'default', 'train_op_parallel', 'multi_thread', 'multi_thread_v2'
train_parallel_mode='default'
num_parallel=4
# 'exponential_decay', 'fasttext_decay', 'polynomial_decay', 'none'
sgd_lr_decay_type='fasttext_decay'
sgd_lr_decay_steps=7600
sgd_lr_decay_rate=0.95
sgd_lr_decay_end_learning_rate=0.0001
sgd_lr_decay_power=1.0
use_clip_gradients=1
clip_norm=1000.0
num_classes=2
pretrained_weights_path='./pretrained_weights/bvlc_alexnet.npy'
train_layers='fc7,fc8'

python main.py \
    --train_data_path ${train_data_path} \
    --eval_data_path ${eval_data_path} \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --epoch ${epoch} \
    --model_dir ${MODEL_DIR} \
    --export_model_dir ${EXPORT_MODEL_DIR} \
    --prefetch_size ${prefetch_size} \
    --shuffle_size ${shuffle_size} \
    --max_train_steps ${max_train_steps} \
    --save_summary_steps ${save_summary_steps} \
    --save_checkpoints_secs ${save_checkpoints_secs} \
    --keep_checkpoint_max 2 \
    --log_step_count_steps ${log_step_count_steps} \
    --use_profile_hook ${use_profile_hook} \
    --profile_steps ${profile_steps} \
    --remove_model_dir ${remove_model_dir} \
    --dropout ${dropout} \
    --shuffle_batch ${shuffle_batch} \
    --optimizer_type ${optimizer_type} \
    --map_num_parallel_calls ${map_num_parallel_calls} \
    --num_classes ${num_classes} \
    --pretrained_weights_path ${pretrained_weights_path} \
    --train_layers ${train_layers}
