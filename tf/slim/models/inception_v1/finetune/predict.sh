#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir

predict_data_path=`pwd`/test.txt
predict_output=`pwd`/predict_output.txt
predict_checkpoint_path=''
lr=0.001
batch_size=64
epoch=30
prefetch_size=500
shuffle_size=500
max_train_steps=-1
save_summary_steps=10
save_checkpoints_secs=600
save_checkpoints_steps=1500
keep_checkpoint_max=5
log_step_count_steps=10
use_profile_hook=0
profile_steps=100
remove_model_dir=1
dropout=0.5
shuffle_batch=1
map_num_parallel_calls=1
num_classes=2
pretrained_weights_path=`pwd`/../pretrained_weights/vgg19.npy
train_layers='fc6,fc7,fc8'
use_data_augmentation=0
optimizer_momentum_momentum=0.9
optimizer_momentum_use_nesterov=0 # bool value
multi_scale_predict=0
inference_shape='256,256'
preprocess_type='easy'  # 'easy', 'vgg'
min_accuracy_increase=0.0001
resize_side_min=256
resize_side_max=256
lr_decay_rate=0.1
lr_decay_epoch_when_no_increase=1
l2_regularizer=0.00001

python common/predict_main.py \
    --predict_data_path ${predict_data_path} \
    --predict_output ${predict_output} \
    --predict_checkpoint_path "${predict_checkpoint_path}" \
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
    --multi_scale_predict ${multi_scale_predict} \
    --inference_shape "${inference_shape}" \
    --preprocess_type ${preprocess_type} \
    --min_accuracy_increase ${min_accuracy_increase} \
    --resize_side_min ${resize_side_min} \
    --resize_side_max ${resize_side_max} \
    --lr_decay_rate ${lr_decay_rate} \
    --lr_decay_epoch_when_no_increase ${lr_decay_epoch_when_no_increase}
