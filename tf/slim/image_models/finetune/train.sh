#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir
model_name='resnet_v1_50'

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
[predict_data_path]=`pwd`/test.txt \
[predict_output]=`pwd`/predict_output.txt \
[predict_checkpoint_path]=${model_dir} \
 \
# train flags \
[batch_size]=64 \
[max_train_steps]=-1 \
[epoch]=10 \
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
[learning_rate_decay_factor]=0.94 \
[num_epochs_per_decay]=2.0 \
[label_smoothing]=0.0 \
 \
# moving average flags
[use_moving_average]=True \
[moving_average_decay]=0.9 \
 \
# preprocess flags \
[inference_image_size]=256 \
[train_image_size]=224 \
[resize_side_min]=256 \
[resize_side_max]=512 \
 \
# finetune flags \
[num_classes]=2 \
[model_name]=${model_name} \
[preprocess_name]=${model_name} \
[pretrained_weights_path]=`pwd`/../pretrained_checkpoint/${model_name}.ckpt \
[trainable_scopes]='resnet_v1_50/logits/' \
[exclude_restore_scopes]='resnet_v1_50/logits/' \
[dropout_keep_prob]=0.8 \
[weight_decay]=0.00005 \
[use_batch_norm]=True \
[batch_norm_decay]=0.9 \
[batch_norm_epsilon]=0.0001 \
[global_pool]=True \
[min_depth]=16 \
[depth_multiplier]=1.0 \
[spatial_squeeze]=True \
[create_aux_logits]=False \
)

params_str=''
for key in $(echo ${!params[*]})
do
    params_str=${params_str}" --"${key}"="${params[$key]}
done
echo 'params: ' ${params_str}

python main.py ${params_str}
