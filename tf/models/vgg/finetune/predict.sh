#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir

predict_data_path=`pwd`/test.txt
predict_output=`pwd`/predict_output.txt
batch_size=128
prefetch_size=1000
shuffle_size=1000
max_train_steps=-1
log_step_count_steps=100
shuffle_batch=1
map_num_parallel_calls=1
num_classes=2
pretrained_weights_path=`pwd`/pretrained_weights/vgg19.npy
use_data_augmentation=0
multi_scale_predict=0
inference_shape='256,256'


python predict_main.py \
    --predict_data_path ${predict_data_path} \
    --predict_output ${predict_output} \
    --batch_size ${batch_size} \
    --model_dir ${model_dir} \
    --export_model_dir ${export_model_dir} \
    --prefetch_size ${prefetch_size} \
    --shuffle_size ${shuffle_size} \
    --max_train_steps ${max_train_steps} \
    --log_step_count_steps ${log_step_count_steps} \
    --shuffle_batch ${shuffle_batch} \
    --map_num_parallel_calls ${map_num_parallel_calls} \
    --num_classes ${num_classes} \
    --pretrained_weights_path ${pretrained_weights_path} \
    --use_data_augmentation ${use_data_augmentation} \
    --multi_scale_predict ${multi_scale_predict} \
    --inference_shape "${inference_shape}"
