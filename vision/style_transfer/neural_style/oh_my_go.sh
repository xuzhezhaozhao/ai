#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

model_dir=model_dir

declare -A params
params=(\
[model_dir]=${model_dir}
[vgg19_npy_path]=../../classification/vgg/pretrained_weights/vgg19.npy
[style_image_path]=./examples/1-style.jpg
[content_image_path]=./examples/1-content.jpg
[output_image_path]=output.jpg
[use_init_image]=false
[init_image_path]=./examples/1-content.jpg
# train flags
[iters]=1000
[learning_rate]=10.0
[adam_beta1]=0.9
[adam_beta2]=0.999
[epsilon]=1e-8
[save_output_steps]=100
[save_summary_steps]=20
[save_checkpoints_steps]=200
[content_loss_weight]=0.01
[style_loss_weight]=1.0
# default: conv4_2,conv5_2
[content_layers]=conv4_2,conv5_2
# default: conv1_1,conv2_1,conv3_1,conv4_1,conv5_1
[style_layers]=conv1_1,conv2_1,conv3_1,conv4_1,conv5_1
[content_layer_loss_weights]=1.0,1.0
[style_layer_loss_weights]=1.0,1.0,1.0,1.0,1.0
)

params_str=''
for key in $(echo ${!params[*]})
do
    params_str=${params_str}" --"${key}"="${params[$key]}
done
echo 'params: ' ${params_str}

python main.py ${params_str}
