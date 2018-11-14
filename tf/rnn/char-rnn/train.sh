#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir

run_mode='train'

remove_model_dir=1
if [[ ${remove_model_dir} == '1' && ${run_mode} != 'predict' ]]; then
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
[train_data_path]=`pwd`/data/tinyshakespeare/input.txt \
[eval_data_path]=`pwd`/data/tinyshakespeare/input.txt \
[predict_data_path]=`pwd`/predict.txt \
[predict_output]=`pwd`/predict_output.${model_name}.txt \
[predict_checkpoint_path]=${model_dir} \
 \
# train flags \
[seq_length]=64 \
[hidden_size]=100 \
[num_layers]=2 \
[keep_prob]=1.0 \
[use_clip_gradients]=True \
[clip_norm]=5.0 \
[batch_size]=32 \
[max_train_steps]=-1 \
[epoch]=200 \
[throttle_secs]=600 \
[use_embedding]=False \
[embedding_dim]=100 \
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
[opt_epsilon]=1e-8 \
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
[learning_rate_decay_type]='fixed' \
[end_learning_rate]=0.0001 \
[learning_rate_decay_factor]=0.95 \
[num_epochs_per_decay]=2.0 \
)

params_str=''
for key in $(echo ${!params[*]})
do
    params_str=${params_str}" --"${key}"="${params[$key]}
done
echo 'params: ' ${params_str}

python main.py ${params_str}
