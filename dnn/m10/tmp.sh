#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

parallel=47

input=raw_data/data.in
sorted_file=${input}.sorted

xcbow_data_dir=xcbow/data
final_data_dir=xcbow/data
preprocessed=${xcbow_data_dir}/data.in.preprocessed
mkdir -p ${xcbow_data_dir}

echo "transform sorted file to fastText format ..."
min_count=50
user_min_watched=10
user_max_watched=512
user_abnormal_watched_thr=2048
user_effective_watched_time_thr=20
user_effective_watched_ratio_thr=0.6
ban_algo_ids='3323,3321,3313,3312,3311,3310,3309,3308,3307,3306,3305,3304,3303,3302,3301'
ban_algo_watched_ratio_thr=1.0
video_play_ratio_bias=30
supress_hot_arg1=-1
supress_hot_arg2=3




mkdir -p log
echo "fastText train ..."
fast_model=${preprocessed}.xcbow
minCount=${min_count}
lr=0.025
dim=100
ws=10
epoch=5
neg=5
bucket=10
minn=0
maxn=0
thread=${parallel}
ts=`date +%Y%m%d%H%M%S`

echo "generate fasttext dict ..."
awk 'NR>2{print $1}' ${fast_model}.vec > ${fast_model}.dict

echo "filter video dict ..."
python filter_dict.py \
        --input_fasttext_dict_file ${fast_model}.dict \
        --input_video_dict_file ${preprocessed}.video_dict \
        --output_fasttext_subset_dict_file ${fast_model}.subset

python filter_vec.py \
        --input_fasttext_vec_file ${fast_model}.output \
        --input_fasttext_subset_dict_file ${fast_model}.subset \
        --output_fasttext_subset_vec_file ${fast_model}.subset.vec

/data/utils/lsh_build_tree ${fast_model}.subset.vec ${fast_model}.subset.lsh 200
