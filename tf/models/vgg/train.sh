#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir

train_data_path=`pwd`/train.txt
eval_data_path=`pwd`/eval.txt
lr=0.001
batch_size=128
epoch=5
prefetch_size=500
shuffle_size=500
max_train_steps=-1
save_summary_steps=50
save_checkpoints_secs=600
save_checkpoints_steps=170
keep_checkpoint_max=2
log_step_count_steps=10
use_profile_hook=0
profile_steps=100
remove_model_dir=1
dropout=0.0
shuffle_batch=1
map_num_parallel_calls=1
num_classes=2
pretrained_weights_path=`pwd`/pretrained_weights/vgg19.npy
train_layers='fc8'
use_data_augmentation=0
optimizer_momentum_momentum=0.9
optimizer_momentum_use_nesterov=0 # bool value
optimizer_exponential_decay_steps=40
optimizer_exponential_decay_rate=0.5
optimizer_exponential_decay_staircase=0  # bool value

if [[ ${remove_model_dir} == '1' ]]; then
    echo "remove model_dir ..."
    rm -rf ${model_dir}.bak
    if [[ -d ${model_dir} ]]; then
        mv ${model_dir} ${model_dir}.bak
    fi
fi

python main.py \
    --train_data_path ${train_data_path} \
    --eval_data_path ${eval_data_path} \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --epoch ${epoch} \
    --model_dir ${model_dir} \
    --export_model_dir ${export_model_dir} \
    --prefetch_size ${prefetch_size} \
    --shuffle_size ${shuffle_size} \
    --max_train_steps ${max_train_steps} \
    --save_summary_steps ${save_summary_steps} \
    --save_checkpoints_secs ${save_checkpoints_secs} \
    --save_checkpoints_steps ${save_checkpoints_steps} \
    --keep_checkpoint_max ${keep_checkpoint_max} \
    --log_step_count_steps ${log_step_count_steps} \
    --use_profile_hook ${use_profile_hook} \
    --profile_steps ${profile_steps} \
    --remove_model_dir ${remove_model_dir} \
    --dropout ${dropout} \
    --shuffle_batch ${shuffle_batch} \
    --map_num_parallel_calls ${map_num_parallel_calls} \
    --num_classes ${num_classes} \
    --pretrained_weights_path ${pretrained_weights_path} \
    --train_layers ${train_layers} \
    --use_data_augmentation ${use_data_augmentation} \
    --optimizer_momentum_momentum ${optimizer_momentum_momentum} \
    --optimizer_momentum_use_nesterov ${optimizer_momentum_use_nesterov} \
    --optimizer_exponential_decay_steps ${optimizer_exponential_decay_steps} \
    --optimizer_exponential_decay_rate ${optimizer_exponential_decay_rate} \
    --optimizer_exponential_decay_staircase ${optimizer_exponential_decay_staircase}
