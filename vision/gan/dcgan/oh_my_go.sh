#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir

# run_mode: train, eval, sample \
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

datadir=../../../datasets/lsun_bedroom-dataset/
train_data_path=${datadir}/train.txt

declare -A params
params=(\
[model_dir]=${model_dir}
[export_model_dir]=${export_model_dir}
[sample_checkpoint_path]=${model_dir}
[run_mode]=${run_mode}
[train_data_path]=${train_data_path}
# train flags
[batch_size]=64
[max_train_steps]=100000
# dataset flags
[prefetch_size]=500
[shuffle_size]=500
[shuffle_batch]=False
[map_num_parallel_calls]=1
# log flags
[save_summary_steps]=10
[save_checkpoints_steps]=100
[keep_checkpoint_max]=3
[log_step_count_steps]=4
[save_output_steps]=500
# optimize
[learning_rate]=0.0002
[adam_beta1]=0.5
[adam_beta2]=0.999
[opt_epsilon]=1e-8
[img_size]=256
[nz]=100
[ngf]=64
[ndf]=64
[nc]=3
)

params_str=''
for key in $(echo ${!params[*]})
do
    params_str=${params_str}" --"${key}"="${params[$key]}
done
echo 'params: ' ${params_str}

python main.py ${params_str}
