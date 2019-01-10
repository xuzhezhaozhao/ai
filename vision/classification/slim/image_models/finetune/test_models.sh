#! /usr/bin/env bash

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

model_names=(\
    'resnet_v1_50' 'resnet_v1_101' 'resnet_v1_152' \
    'resnet_v2_50' 'resnet_v2_101' 'resnet_v2_152' \
    'inception_v1' 'inception_v2' 'inception_v3' \
    'inception_v4' 'inception_resnet_v2' \
    'vgg_16' 'vgg_19' \
    )

for model_name in ${model_names[@]}
do
    echo "test ${model_name} ..."
    ./train.sh ${model_name} > test_${model_name}.txt 2>&1
    echo "test ${model_name} done"
done
