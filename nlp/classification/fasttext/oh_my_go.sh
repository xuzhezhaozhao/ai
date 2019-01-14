#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

mkdir -p model
#datadir=../../../datasets/kd_video_comments-dataset/data/fasttext
datadir=../../../datasets/thucnews-dataset/data/fasttext

../../../submodules/fastText/fasttext \
    supervised \
    -input ${datadir}/train.txt \
    -output model/model \
    -dim 100 \
    -lr 0.025 \
    -wordNgrams 2 \
    -minCount 10 \
    -bucket 2000000 \
    -epoch 20 \
    -thread 7

echo 'train accuracy:'
../../../submodules/fastText/fasttext \
    test \
    model/model.bin \
    ${datadir}/train.txt

echo 'test accuracy:'
../../../submodules/fastText/fasttext \
    test \
    model/model.bin \
    ${datadir}/test.txt

../../../submodules/fastText/fasttext \
    predict \
    model/model.bin \
    ${datadir}/test.txt > preidct.txt

python check_error.py preidct.txt ${datadir}/test.txt > error.txt
rm preidct.txt
