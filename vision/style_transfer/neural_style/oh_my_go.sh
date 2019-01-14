#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

model_dir=model_dir

declare -A params
params=(\
[model_dir]=${model_dir} \
[vgg19_npy_path]=../../classification/vgg/pretrained_weights/vgg19.npy \
 \
[style_image_path]=./examples/1-style.jpg \
[content_image_path]=./examples/1-content.jpg \
[output_image_path]=output.jpg \
[use_init_image]=false \
[init_image_path]=./examples/1-content.jpg \
 \
# train flags \
[iters]=1000 \
)

params_str=''
for key in $(echo ${!params[*]})
do
    params_str=${params_str}" --"${key}"="${params[$key]}
done
echo 'params: ' ${params_str}

python main.py ${params_str}
