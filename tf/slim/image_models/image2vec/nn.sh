#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

python nn.py \
    --dict test.txt \
    --image_features features.txt \
    --output nn.txt \
    --k 100
