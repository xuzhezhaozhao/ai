#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir

train_data_path=`pwd`/train.txt
eval_data_path=`pwd`/validation.txt
predict_data_path=`pwd`/test.txt
predict_output=`pwd`/predict_output.txt
lr=0.001
batch_size=128
epoch=5
prefetch_size=1000
shuffle_size=1000
max_train_steps=-1
save_summary_steps=50
save_checkpoints_secs=600
keep_checkpoint_max=2
log_step_count_steps=100
use_profile_hook=0
profile_steps=100
remove_model_dir=1
dropout=0.5
shuffle_batch=1
map_num_parallel_calls=1
num_classes=2
pretrained_weights_path=`pwd`/pretrained_weights/bvlc_alexnet.npy
train_layers='fc6,fc7,fc8'
use_data_augmentation=0


python predict_main.py \
    --train_data_path ${train_data_path} \
    --eval_data_path ${eval_data_path} \
    --predict_data_path ${predict_data_path} \
    --predict_output ${predict_output} \
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
    --use_data_augmentation ${use_data_augmentation}
