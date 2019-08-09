#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

mkdir -p model
#datadir=../../../datasets/thucnews-dataset/data/fasttext
datadir=../../../datasets/kd_video_comments-dataset/data/fasttext/
train_data=${datadir}/train.txt
test_data=${datadir}/test.txt

../../../submodules/fastText/fasttext \
    supervised \
    -input ${train_data} \
    -output model/model \
    -dim 100 \
    -lr 0.100 \
    -wordNgrams 2 \
    -minCount 10 \
    -bucket 2000000 \
    -epoch 20 \
    -thread 7

echo 'train accuracy:'
../../../submodules/fastText/fasttext \
    test \
    model/model.bin \
    ${train_data}

echo 'test accuracy:'
../../../submodules/fastText/fasttext \
    test \
    model/model.bin \
    ${test_data}

../../../submodules/fastText/fasttext \
    predict \
    model/model.bin \
    ${test_data} > preidct.txt

python check_error.py preidct.txt ${test_data} > error.txt
rm preidct.txt
