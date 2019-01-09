#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

mkdir -p word2vec_model_dir
datadir=../../../datasets/kd_video_comments-dataset/data/fasttext

lr=0.025
ws=5
epoch=20
minCount=5
neg=10
bucket=2000000
minn=0
maxn=0
thread=7
t=1e-4
../../../submodules/fastText/fasttext \
    skipgram \
    -input ${datadir}/train.txt \
    -output word2vec_model_dir/word2vec \
	-lr ${lr} \
    -dim 50 \
	-ws ${ws} \
	-epoch ${epoch} \
	-minCount ${minCount} \
    -minCountLabel 2 \
	-neg ${neg} \
	-loss ns \
	-bucket ${bucket} \
  	-minn ${minn} \
	-maxn ${maxn} \
	-thread ${thread} \
	-t ${t} \
	-lrUpdateRate 100

awk 'NR>1{print $1}' word2vec_model_dir/word2vec.vec > word2vec_model_dir/word2vec.dict
awk 'NR>1{print $1}' ${datadir}/train.txt | sort | uniq > word2vec_model_dir/label.dict
