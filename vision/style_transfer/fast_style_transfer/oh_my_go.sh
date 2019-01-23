#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir

# run_mode: train, eval, predict \
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

datadir=../../../datasets/coco2014-dataset/
train_data_path=${datadir}/train.txt
eval_data_path=${datadir}/test.txt

declare -A params
params=(\
[model_dir]=${model_dir}
[export_model_dir]=${export_model_dir}
[run_mode]=${run_mode}
[train_data_path]=${train_data_path}
[eval_data_path]=${eval_data_path}
[predict_image_path]=../examples/content/stata.jpg
[predict_checkpoint_path]=${model_dir}
[vgg19_npy_path]=../../classification/vgg/pretrained_weights/vgg19.npy
[style_image_path]=../examples/style/stars.jpg
[content_loss_weight]=10
[style_loss_weight]=200.0
[total_variation_loss_weight]=0.1
# default: conv4_2,conv5_2
[content_layers]=conv4_2
# default: conv1_1,conv2_1,conv3_1,conv4_1,conv5_1
[style_layers]=conv1_1,conv2_1,conv3_1,conv4_1,conv5_1
[content_layer_loss_weights]=1.0
[style_layer_loss_weights]=1.0
# train flags
[batch_size]=4
[eval_batch_size]=4
[max_train_steps]=-1
[epoch]=2
[throttle_secs]=60
# dataset flags
[prefetch_size]=50
[shuffle_size]=50
[shuffle_batch]=False
[map_num_parallel_calls]=1
# log flags
[save_summary_steps]=10
[save_checkpoints_secs]=900
[save_checkpoints_steps]=-1
[keep_checkpoint_max]=3
[log_step_count_steps]=4
# optimizer flags
[optimizer]='adam'
[adadelta_rho]=0.95
[adagrad_init_value]=0.1
[adam_beta1]=0.9
[adam_beta2]=0.999
[opt_epsilon]=1e-8
[ftrl_learning_rate_power]=-0.5
[ftrl_initial_accumulator_value]=0.1
[ftrl_l1]=0.0
[ftrl_l2]=0.0
[momentum]=0.9
[rmsprop_momentum]=0.0
[rmsprop_decay]=0.9
# learning rate flags
[learning_rate]=0.001
## fixed, exponential or polynomial
[learning_rate_decay_type]='fixed'
[end_learning_rate]=0.0001
[learning_rate_decay_factor]=0.9
[decay_steps]=100
)

params_str=''
for key in $(echo ${!params[*]})
do
    params_str=${params_str}" --"${key}"="${params[$key]}
done
echo 'params: ' ${params_str}

python main.py ${params_str}
