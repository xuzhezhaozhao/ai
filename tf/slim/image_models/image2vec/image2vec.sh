#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

# image path file, one per line
input='test.txt'

# image features, one per line
output='features.txt'

# batch size, as large as the machine can handle
batch_size=32

# one of 'resnet_v2_50', 'inception'
model_name='resnet_v2_50'

# one of 'inception', 'vgg'
# suggested config:
#   resnet_v2_50: inception, 299
#   inception_v3: inception, 299
preprocessing_name='inception'
image_size=299

python image2vec.py \
    --input ${input} \
    --output ${output} \
    --batch_size ${batch_size} \
    --model_name ${model_name} \
    --preprocessing_name ${preprocessing_name} \
    --image_size ${image_size}
