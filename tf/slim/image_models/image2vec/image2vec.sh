#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

input='test.txt'
output='features.txt'
batch_size=32
model_name=resnet_v2_50
preprocessing_name=inception
image_size=299

python image2vec.py \
    --input ${input} \
    --output ${output} \
    --batch_size ${batch_size} \
    --model_name ${model_name} \
    --preprocessing_name ${preprocessing_name} \
    --image_size ${image_size}
