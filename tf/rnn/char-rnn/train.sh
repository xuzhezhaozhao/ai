#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir
start_string=""
train_data_path=./ref2/data/linux.txt
use_embedding=False
seq_length=400
num_layers=3

run_mode='sample'
if [[ $# -eq 1 ]]; then
    run_mode=$1
fi

remove_model_dir=1
if [[ ${remove_model_dir} == '1' && ${run_mode} != 'sample' ]]; then
    echo "remove ${model_dir}.bak"
    rm -rf ${model_dir}.bak
    if [[ -d ${model_dir} ]]; then
        echo "rename ${model_dir} to ${model_dir}.bak"
        mv ${model_dir} ${model_dir}.bak
    fi
fi

mkdir -p ${model_dir}

declare -A params
params=(\
[model_dir]=${model_dir} \
[export_model_dir]=${export_model_dir} \
[preprocessed_filename]=${model_dir}/preprocessed.pkl \
 \
## run_mode: train, predict, all \
[run_mode]=${run_mode} \
[train_data_path]=${train_data_path} \
[start_string]=${start_string} \
[num_samples]=2000 \
 \
# train flags \
[seq_length]=${seq_length} \
[hidden_size]=128 \
[num_layers]=${num_layers} \
[keep_prob]=0.6 \
[use_clip_gradients]=True \
[clip_norm]=5.0 \
[batch_size]=32 \
[epoch]=20 \
[use_embedding]=${use_embedding} \
[embedding_dim]=128 \
[min_count]=2 \
[sample_top_k]=5 \
 \
# log flags \
[save_summary_steps]=10 \
[save_checkpoints_steps]=1000 \
[keep_checkpoint_max]=10 \
[log_step_count_steps]=100 \
 \
# optimizer flags \
[optimizer]='rmsprop' \
[adadelta_rho]=0.95 \
[adagrad_initial_accumulator_value]=0.1 \
[adam_beta1]=0.9 \
[adam_beta2]=0.999 \
[opt_epsilon]=1e-8 \
[ftrl_learning_rate_power]=-0.5 \
[ftrl_initial_accumulator_value]=0.1 \
[ftrl_l1]=0.0 \
[ftrl_l2]=0.0 \
[momentum]=0.95 \
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
)

params_str=''
for key in $(echo ${!params[*]})
do
    params_str=${params_str}" --"${key}"="${params[$key]}
done
echo 'params: ' ${params_str}

python main.py ${params_str} --start_string "${start_string}"
