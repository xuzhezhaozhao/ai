#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

raw_data_dir=tmp_hdfs/data1
data_dir=data1_ing
final_data_dir=data1
input=${data_dir}/data.in
sorted_file=${input}.sorted

parallel=47

echo "transform sorted file ..."
user_min_watched=10
user_max_watched=512
user_abnormal_watched_thr=2048
user_effective_watched_time_thr=20
user_effective_watched_ratio_thr=0.3
min_count=50
ban_algo_ids='3323,3321,3313,3312,3311,3310,3309,3308,3307,3306,3305,3304,3303,3302,3301'
ban_algo_watched_ratio_thr=0.8
video_play_ratio_bias=5

preprocessed=${input}.preprocessed


echo "fastText train ..."
fast_model=${preprocessed}.shuf
minCount=${min_count}
lr=0.025
dim=100
ws=15
epoch=5
neg=5
bucket=2000000
minn=0
maxn=0
thread=${parallel}
ts=`date +%Y%m%d%H%M%S`

# call newman's disable interface
/data/utils/enable_filter -input ${fast_model}.dict -output_valid ${fast_model}.dict.valid -output_filtered ${fast_model}.dict.filtered

echo "fasttext nn ..."
nn_k=100
FASTTEST=/data/utils/fasttext
#${FASTTEST} multi-nnsubset ${fast_model}.bin ${fast_model}.dict ${fast_model}.dict.valid ${parallel} ${nn_k}
${FASTTEST} multi-nn ${fast_model}.bin ${fast_model}.dict ${parallel} ${nn_k}
mv ${fast_model}.dict.result ${fast_model}.result.raw

rm -rf ${final_data_dir}.bak
if [ -d ${final_data_dir} ]; then
    mv ${final_data_dir} ${final_data_dir}.bak
fi
mv ${data_dir} ${final_data_dir}
