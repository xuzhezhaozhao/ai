#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir
model_name='inception_v3'
if [[ $# -eq 1 ]]; then
    model_name=$1
fi
pretrained_weights_path=`pwd`/../pretrained_checkpoint/${model_name}.ckpt

# default preprocess image flags
preprocess_name='vgg'
eval_image_size=224
train_image_size=224
resize_side_min=256
resize_side_max=512
train_using_one_crop=False
global_pool=False
create_aux_logits=False

# modified by specified model
preprocess_image() {
    if [[ $1 == 'resnet_v1_50' ]]; then
        trainable_scopes='resnet_v1_50/logits/'
        exclude_restore_scopes='resnet_v1_50/logits/,global_step:0'
        preprocess_name='vgg'
        eval_image_size=256
        train_image_size=224
        resize_side_min=256
        resize_side_max=352
    elif [[ $1 == 'resnet_v1_101' ]]; then
        trainable_scopes='resnet_v1_101/logits/'
        exclude_restore_scopes='resnet_v1_101/logits/,global_step:0'
        preprocess_name='vgg'
        eval_image_size=256
        train_image_size=224
        resize_side_min=256
        resize_side_max=352
    elif [[ $1 == 'resnet_v1_152' ]]; then
        trainable_scopes='resnet_v1_152/logits/'
        exclude_restore_scopes='resnet_v1_152/logits/,global_step:0'
        preprocess_name='vgg'
        eval_image_size=256
        train_image_size=224
        resize_side_min=256
        resize_side_max=352
    elif [[ $1 == 'resnet_v2_50' ]]; then
        trainable_scopes='resnet_v2_50/logits/'
        exclude_restore_scopes='resnet_v2_50/logits/,global_step:0'
        preprocess_name='inception'
        eval_image_size=356
        train_image_size=299
        resize_side_min=299
        resize_side_max=352
    elif [[ $1 == 'resnet_v2_101' ]]; then
        trainable_scopes='resnet_v2_101/logits/'
        exclude_restore_scopes='resnet_v2_101/logits/,global_step:0'
        preprocess_name='inception'
        eval_image_size=356
        train_image_size=299
        resize_side_min=299
        resize_side_max=352
    elif [[ $1 == 'resnet_v2_152' ]]; then
        trainable_scopes='resnet_v2_152/logits/'
        exclude_restore_scopes='resnet_v2_152/logits/,global_step:0'
        preprocess_name='inception'
        eval_image_size=356
        train_image_size=299
        resize_side_min=299
        resize_side_max=352
    elif [[ $1 == 'inception_v1' ]]; then
        trainable_scopes='InceptionV1/Logits/'
        exclude_restore_scopes='InceptionV1/Logits/,global_step:0'
        preprocess_name='inception'
        eval_image_size=256
        train_image_size=224
    elif [[ $1 == 'inception_v2' ]]; then
        trainable_scopes='InceptionV2/Logits/'
        exclude_restore_scopes='InceptionV2/Logits/,global_step:0'
        preprocess_name='inception'
        eval_image_size=256
        train_image_size=224
    elif [[ $1 == 'inception_v3' ]]; then
        trainable_scopes='InceptionV3/Logits/'
        exclude_restore_scopes='InceptionV3/Logits/,global_step:0'
        preprocess_name='inception'
        eval_image_size=356
        train_image_size=299
    elif [[ $1 == 'inception_v4' ]]; then
        trainable_scopes='InceptionV4/Logits/'
        exclude_restore_scopes='InceptionV4/Logits/,global_step:0'
        preprocess_name='inception'
        eval_image_size=356
        train_image_size=299
    elif [[ $1 == 'inception_resnet_v2' ]]; then
        trainable_scopes='InceptionResnetV2/Logits/'
        exclude_restore_scopes='InceptionResnetV2/Logits/,global_step:0'
        preprocess_name='inception'
        eval_image_size=356
        train_image_size=299
    elif [[ $1 == 'vgg_16' ]]; then
        trainable_scopes='vgg_16/fc6/,vgg_16/fc7/,vgg_16/fc8/'
        exclude_restore_scopes='vgg_16/fc8/,global_step:0'
        preprocess_name='vgg'
        eval_image_size=256
        train_image_size=224
        resize_side_min=256
        resize_side_max=512
    elif [[ $1 == 'vgg_19' ]]; then
        trainable_scopes='vgg_19/fc6/,vgg_19/fc7/,vgg_19/fc8/'
        exclude_restore_scopes='vgg_19/fc8/,global_step:0'
        preprocess_name='vgg'
        eval_image_size=256
        train_image_size=224
        resize_side_min=256
        resize_side_max=512
    elif [[ $1 == 'mobilenet_v1_0.25_128' ]]; then
        pretrained_weights_path=${pretrained_weights_path}/${model_name}.ckpt
        trainable_scopes='MobilenetV1/Logits/'
        exclude_restore_scopes='MobilenetV1/Logits/,global_step:0'
        preprocess_name='inception'
        eval_image_size=128
        train_image_size=128
    elif [[ $1 == 'mobilenet_v1_0.5_160' ]]; then
        pretrained_weights_path=${pretrained_weights_path}/${model_name}.ckpt
        trainable_scopes='MobilenetV1/Logits/'
        exclude_restore_scopes='MobilenetV1/Logits/,global_step:0'
        preprocess_name='inception'
        eval_image_size=160
        train_image_size=160
    elif [[ $1 == 'mobilenet_v1_1.0_224' ]]; then
        pretrained_weights_path=${pretrained_weights_path}/${model_name}.ckpt
        trainable_scopes='MobilenetV1/Logits/'
        exclude_restore_scopes='MobilenetV1/Logits/,global_step:0'
        preprocess_name='inception'
        eval_image_size=224
        train_image_size=224
    elif [[ $1 == 'mobilenet_v2_1.0_224' ]]; then
        pretrained_weights_path=${pretrained_weights_path}/${model_name}.ckpt
        trainable_scopes='MobilenetV2/Logits/'
        exclude_restore_scopes='MobilenetV2/Logits/,global_step:0'
        preprocess_name='inception'
        eval_image_size=224
        train_image_size=224
    elif [[ $1 == 'mobilenet_v2_1.4_224' ]]; then
        pretrained_weights_path=${pretrained_weights_path}/${model_name}.ckpt
        trainable_scopes='MobilenetV2/Logits/'
        exclude_restore_scopes='MobilenetV2/Logits/,global_step:0'
        preprocess_name='inception'
        eval_image_size=224
        train_image_size=224
    elif [[ $1 == 'nasnet_mobile' ]]; then
        pretrained_weights_path=${pretrained_weights_path}/model.ckpt
        trainable_scopes='final_layer/FC'
        exclude_restore_scopes='final_layer/FC,global_step:0'
        preprocess_name='inception'
        eval_image_size=224
        train_image_size=224
    elif [[ $1 == 'nasnet_large' ]]; then
        pretrained_weights_path=${pretrained_weights_path}/model.ckpt
        trainable_scopes='final_layer/FC'
        exclude_restore_scopes='final_layer/FC,global_step:0'
        preprocess_name='inception'
        eval_image_size=224
        train_image_size=224
    elif [[ $1 == 'pnasnet_mobile' ]]; then
        pretrained_weights_path=${pretrained_weights_path}/model.ckpt
        trainable_scopes='final_layer/FC'
        exclude_restore_scopes='final_layer/FC,global_step:0'
        preprocess_name='inception'
        eval_image_size=224
        train_image_size=224
    elif [[ $1 == 'pnasnet_large' ]]; then
        pretrained_weights_path=${pretrained_weights_path}/model.ckpt
        trainable_scopes='final_layer/FC'
        exclude_restore_scopes='final_layer/FC,global_step:0'
        preprocess_name='inception'
        eval_image_size=224
        train_image_size=224
    fi
}

preprocess_image ${model_name}

remove_model_dir=1
if [[ ${remove_model_dir} == '1' ]]; then
    echo "remove ${model_dir}.bak"
    rm -rf ${model_dir}.bak
    if [[ -d ${model_dir} ]]; then
        echo "rename ${model_dir} to ${model_dir}.bak"
        mv ${model_dir} ${model_dir}.bak
    fi
fi

declare -A params
params=(\
[model_dir]=${model_dir} \
[export_model_dir]=${export_model_dir} \
 \
## run_mode: train, predict, all \
[run_mode]='all' \
[train_data_path]=`pwd`/train.txt \
[eval_data_path]=`pwd`/validation.txt \
[predict_data_path]=`pwd`/validation.txt \
[predict_output]=`pwd`/predict_output.${model_name}.txt \
[predict_checkpoint_path]=${model_dir} \
 \
# train flags \
[batch_size]=32 \
[max_train_steps]=-1 \
[epoch]=5 \
[throttle_secs]=60 \
 \
# dataset flags \
[prefetch_size]=500 \
[shuffle_size]=500 \
[shuffle_batch]=True \
[map_num_parallel_calls]=1 \
 \
# log flags \
[save_summary_steps]=10 \
[save_checkpoints_secs]=-1 \
[save_checkpoints_steps]=-1 \
[save_checkpoints_epoches]=1.0 \
[keep_checkpoint_max]=5 \
[log_step_count_steps]=10 \
 \
# profile flags \
[use_profile_hook]=False \
[profile_steps]=100 \
 \
# optimizer flags \
[optimizer]='momentum' \
[adadelta_rho]=0.95 \
[adagrad_initial_accumulator_value]=0.1 \
[adam_beta1]=0.9 \
[adam_beta2]=0.999 \
[opt_epsilon]=1.0 \
[ftrl_learning_rate_power]=-0.5 \
[ftrl_initial_accumulator_value]=0.1 \
[ftrl_l1]=0.0 \
[ftrl_l2]=0.0 \
[momentum]=0.9 \
[rmsprop_momentum]=0.9 \
[rmsprop_decay]=0.9 \
 \
# learning rate flags \
[learning_rate]=0.001 \
## fixed, exponential or polynomial
[learning_rate_decay_type]='exponential' \
[end_learning_rate]=0.0001 \
[learning_rate_decay_factor]=0.9 \
[num_epochs_per_decay]=1.0 \
[label_smoothing]=0.0 \
 \
# moving average flags
[use_moving_average]=False \
[moving_average_decay]=0.9 \
 \
# preprocess flags \
[eval_image_size]=${eval_image_size} \
[train_image_size]=${train_image_size} \
[resize_side_min]=${resize_side_min} \
[resize_side_max]=${resize_side_max} \
[train_using_one_crop]=${train_using_one_crop} \
 \
# finetune flags \
[num_classes]=2 \
[model_name]=${model_name} \
[preprocess_name]=${preprocess_name} \
[pretrained_weights_path]=${pretrained_weights_path} \
[trainable_scopes]=${trainable_scopes} \
[exclude_restore_scopes]=${exclude_restore_scopes} \
[dropout_keep_prob]=0.8 \
[weight_decay]=0.0000 \
[use_batch_norm]=True \
[batch_norm_decay]=0.9 \
[batch_norm_epsilon]=0.0001 \
[global_pool]=${global_pool} \
[min_depth]=16 \
[depth_multiplier]=1.0 \
[spatial_squeeze]=False \
[create_aux_logits]=${create_aux_logits} \
)

params_str=''
for key in $(echo ${!params[*]})
do
    params_str=${params_str}" --"${key}"="${params[$key]}
done
echo 'params: ' ${params_str}

python main.py ${params_str}
