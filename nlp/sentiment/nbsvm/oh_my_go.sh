#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

mkdir -p nbsvm_run
cd nbsvm_run

if [ ! -d liblinear-1.96 ]; then
wget https://www.csie.ntu.edu.tw/~cjlin/liblinear/oldfiles/liblinear-1.96.zip
unzip liblinear-1.96.zip
rm liblinear-1.96.zip
cd liblinear-1.96
make
cd ..
fi

datadir=../../../../datasets/kd_video_comments-dataset/data/nbsvm
python ../nbsvm.py \
    --liblinear liblinear-1.96 \
    --ptrain ${datadir}/train-pos.txt \
    --ntrain ${datadir}/train-neg.txt \
    --ptest ${datadir}/test-pos.txt \
    --ntest ${datadir}/test-neg.txt \
    --ngram 12 \
    --out NBSVM-TEST
