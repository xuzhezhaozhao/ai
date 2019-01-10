#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir
word2vec_model_dir=`pwd`/word2vec_model_dir

# run_mode: train, eval, predict, all \
run_mode=train

if [[ $# -eq 1 ]]; then
    run_mode=$1
fi

echo 'run_mode =' ${run_mode}

remove_model_dir=1
if [[ ${run_mode} != 'train' && ${run_mode} != 'all' ]]; then
    remove_model_dir=0
fi
if [[ ${remove_model_dir} == '1' ]]; then
    echo "remove ${model_dir}.bak"
    rm -rf ${model_dir}.bak
    if [[ -d ${model_dir} ]]; then
        echo "rename ${model_dir} to ${model_dir}.bak"
        mv ${model_dir} ${model_dir}.bak
    fi
fi

datadir=../../../datasets/kd_video_comments-dataset/data/char-cnn
train_data_path=${datadir}/train.txt
eval_data_path=${datadir}/test.txt
predict_output=`pwd`/predict.txt

declare -A params
params=(\
[model_dir]=${model_dir} \
[export_model_dir]=${export_model_dir} \
 \
[run_mode]=${run_mode} \
[train_data_path]=${train_data_path} \
[eval_data_path]=${eval_data_path} \
[predict_data_path]=${eval_data_path} \
[predict_output]=${predict_output} \
[predict_checkpoint_path]=${model_dir} \
 \
[char_dict_path]=char.dict \
[label_dict_path]=label.dict \
 \
# train flags \
[batch_size]=32 \
[eval_batch_size]=2048 \
[max_train_steps]=-1 \
[epoch]=5 \
[throttle_secs]=60 \
 \
# dataset flags \
[label_str]='__label__' \
[max_length]=80 \
[num_filters]=128 \
[filter_sizes]=2,3,4 \
[label_weights]=1.0,0.5 \
[embedding_dim]=128 \
[prefetch_size]=2000 \
[shuffle_size]=1000 \
[shuffle_batch]=True \
[map_num_parallel_calls]=1 \
 \
# log flags \
[save_summary_steps]=100 \
[save_checkpoints_secs]=900 \
[save_checkpoints_steps]=-1 \
[keep_checkpoint_max]=5 \
[log_step_count_steps]=100 \
 \
# profile flags \
[use_profile_hook]=False \
[profile_steps]=100 \
 \
# optimizer flags \
[optimizer]='adam' \
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
[rmsprop_momentum]=0.0 \
[rmsprop_decay]=0.9 \
 \
# learning rate flags \
[learning_rate]=0.025 \
## fixed, exponential or polynomial
[learning_rate_decay_type]='fixed' \
[end_learning_rate]=0.0001 \
[learning_rate_decay_factor]=0.9 \
[decay_steps]=100 \
 \
# moving average flags
[use_moving_average]=False \
[moving_average_decay]=0.9 \
 \
[dropout_keep_prob]=1.0 \
[weight_decay]=0.0000 \
[use_batch_norm]=True \
[batch_norm_decay]=0.9 \
[batch_norm_epsilon]=0.0001 \
)

params_str=''
for key in $(echo ${!params[*]})
do
    params_str=${params_str}" --"${key}"="${params[$key]}
done
echo 'params: ' ${params_str}

python main.py ${params_str}


if [[ ${run_mode} == 'predict' || ${run_mode} == 'all' ]]; then
    python check_error.py ${predict_output} ${eval_data_path} > error.txt
fi
