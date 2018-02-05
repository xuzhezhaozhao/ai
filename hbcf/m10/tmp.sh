#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

raw_data_dir=tmp_hdfs/data1
data_dir=data1.bak
input=${data_dir}/data.in
sorted_file=${input}.sorted

parallel=47

echo "transform sorted file ..."
min_count=50
user_min_watched=10
user_max_watched=512
user_abnormal_watched_thr=2048
user_effective_watched_time_thr=20
user_effective_watched_ratio_thr=0.3
ban_algo_ids='3323,3321,3313,3312,3311,3310,3309,3308,3307,3306,3305,3304,3303,3302,3301'
ban_algo_watched_ratio_thr=0.8
video_play_ratio_bias=10
supress_hot_arg1=-1
supress_hot_arg2=3

preprocessed=${input}.preprocessed


echo "fastText train ..."
fast_model=${preprocessed}.shuf
minCount=${min_count}
lr=0.025
dim=100
ws=15
epoch=5
neg=5
bucket=10
minn=0
maxn=0
thread=${parallel}
ts=`date +%Y%m%d%H%M%S`

echo "fasttext nn ..."
nn_k=100
FASTTEST=./fasttext
${FASTTEST} multi-nn ${fast_model}.bin ${fast_model}.dict ${parallel} ${nn_k}
mv ${fast_model}.dict.result ${fast_model}.result.raw
