#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir

declare -A params
params=(\
[model_dir]=${model_dir} \
[export_model_dir]=${export_model_dir} \
 \
## run_mode: train, predict, all \
[run_mode]='all' \
[train_data_path]=`pwd`/train.txt \
[eval_data_path]=`pwd`/validation.txt \
[predict_data_path]='pwd'/test.txt \
[predict_output]='pwd'/predict_output.txt \
[predict_checkpoint_path]=${model_dir} \
 \
# train flags \
[lr]=0.001 \
[batch_size]=64 \
[max_train_steps]=-1 \
[epoch]=1 \
[throttle_secs]=300 \
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
[save_checkpoints_steps]=1500 \
[keep_checkpoint_max]=5 \
[log_step_count_steps]=10 \
 \
# profile flags \
[use_profile_hook]=False \
[profile_steps]=100 \
 \
# optimizer flags \
[optimizer_momentum_momentum]=0.9 \
[optimizer_momentum_use_nesterov]=False \
 \
# preprocess flags \
[inference_image_size]=256 \
[train_image_size]=224 \
[resize_side_min]=256 \
[resize_side_max]=512 \
 \
# finetune flags \
[num_classes]=2 \
[model_name]='resnet_v1_50' \
[preprocess_name]='resnet_v1_50' \
[pretrained_weights_path]=`pwd`/../pretrained_checkpoint/resnet_v1_50.ckpt \
[train_layers]='resnet_v1_50/logits/' \
[exclude_restore_layers]='resnet_v1_50/logits/' \
[dropout_keep_prob]=0.5 \
[weights_decay]=0.0001 \
[use_batch_norm]=True \
[batch_norm_decay]=0.9 \
[batch_norm_epsilon]=0.0001 \
[global_pool]=True \
[min_depth]=16 \
[depth_multiplier]=1.0 \
[spatial_squeeze]=True \
[create_aux_logits]=False \
)

remove_model_dir=1
if [[ ${remove_model_dir} == '1' ]]; then
    echo "remove model_dir ..."
    rm -rf ${model_dir}.bak
    if [[ -d ${model_dir} ]]; then
        mv ${model_dir} ${model_dir}.bak
    fi
fi

params_str=''
for key in $(echo ${!params[*]})
do
    params_str=${params_str}" --"${key}"="${params[$key]}
done
echo 'params: ' ${params_str}

python common/main.py ${params_str}
