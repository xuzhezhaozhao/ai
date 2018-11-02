#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir

train_data_path=`pwd`/train.txt
eval_data_path=`pwd`/validation.txt
lr=0.001
batch_size=64
epoch=30
prefetch_size=500
shuffle_size=500
max_train_steps=-1
save_summary_steps=10
save_checkpoints_secs=-1
save_checkpoints_steps=1500
keep_checkpoint_max=5
log_step_count_steps=10
use_profile_hook=0
profile_steps=100
remove_model_dir=1
dropout_keep_prob=0.5
shuffle_batch=1
map_num_parallel_calls=4
num_classes=2
pretrained_weights_path=`pwd`/../pretrained_checkpoint/inception_v1.ckpt
train_layers='InceptionV1/Logits/'
exclude_restore_layers='InceptionV1/Logits/'
use_data_augmentation=0
optimizer_momentum_momentum=0.9
optimizer_momentum_use_nesterov=0 # bool value
multi_scale_predict=0
inference_shape='256,256'
preprocess_type='vgg'  # 'easy', 'vgg'
min_accuracy_increase=0.0001
resize_side_min=256
resize_side_max=256
lr_decay_rate=0.1
lr_decay_epoch_when_no_increase=1
l2_regularizer=0.0000
use_batch_norm=1
batch_norm_decay=0.95
batch_norm_epsilon=0.001
global_pool=0
model_name='inception_v1'
train_image_size=224


if [[ ${remove_model_dir} == '1' ]]; then
    echo "remove model_dir ..."
    rm -rf ${model_dir}.bak
    if [[ -d ${model_dir} ]]; then
        mv ${model_dir} ${model_dir}.bak
    fi
fi

python common/main.py \
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
    --dropout_keep_prob ${dropout_keep_prob} \
    --shuffle_batch ${shuffle_batch} \
    --map_num_parallel_calls ${map_num_parallel_calls} \
    --num_classes ${num_classes} \
    --pretrained_weights_path ${pretrained_weights_path} \
    --train_layers ${train_layers} \
    --exclude_restore_layers "${exclude_restore_layers}" \
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
    --lr_decay_epoch_when_no_increase ${lr_decay_epoch_when_no_increase} \
    --l2_regularizer ${l2_regularizer} \
    --use_batch_norm ${use_batch_norm} \
    --batch_norm_decay ${batch_norm_decay} \
    --batch_norm_epsilon ${batch_norm_epsilon} \
    --global_pool ${global_pool} \
    --model_name ${model_name} \
    --train_image_size ${train_image_size}
