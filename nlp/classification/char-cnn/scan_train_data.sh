#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

datadir=../../../datasets/kd_video_comments-dataset/data/char-cnn
train_data_path=${datadir}/train.txt

python count.py ${train_data_path} 50 > char.dict
awk 'NR>1{print $1}' ${datadir}/train.txt | sort | uniq > label.dict
