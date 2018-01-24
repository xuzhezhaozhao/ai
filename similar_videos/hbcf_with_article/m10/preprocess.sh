#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

mkdir -p log

raw_data_dir=tmp_hdfs/data0
data_dir=data0_ing
final_data_dir=data0
input=${data_dir}/data.in
sorted_file=${input}.sorted

parallel=47

echo "transform sorted file ..."
user_min_watched=20
user_max_watched=512
user_abnormal_watched_thr=2048
user_effective_watched_time_thr=20
user_effective_watched_ratio_thr=0.3
min_count=50
ban_algo_ids='3323,3321,3313,3312,3311,3310,3309,3308,3307,3306,3305,3304,3303,3302,3301'
ban_algo_watched_ratio_thr=0.8
video_play_ratio_bias=30

preprocessed=${input}.preprocessed
/data/preprocess/build/src/preprocess \
	-raw_input=${sorted_file} \
	-with_header=false \
	-only_video=false \
	-interval=1000000 \
	-output_user_watched_file=${preprocessed} \
        -output_user_watched_ratio_file=${preprocessed}.watched_ratio \
        -output_video_play_ratio_file=${preprocessed}.play_ratio \
	-user_min_watched=${user_min_watched} \
	-user_max_watched=${user_max_watched} \
	-user_abnormal_watched_thr=${user_abnormal_watched_thr} \
	-supress_hot_arg1=20 \
	-supress_hot_arg2=3 \
        -user_effective_watched_time_thr=${user_effective_watched_time_thr} \
        -user_effective_watched_ratio_thr=${user_effective_watched_ratio_thr} \
        -min_count=${min_count} \
        -ban_algo_watched_ratio_thr=${ban_algo_watched_ratio_thr} \
        -ban_algo_ids=${ban_algo_ids} \
        -video_play_ratio_bias=${video_play_ratio_bias} \
        -output_video_dict_file=${preprocessed}.video_dict \
        -output_article_dict_file=${preprocessed}.article_dict \
        -output_video_click_file=${preprocessed}.video_click \
        -ban_unknow_algo_id=false


echo "fastText train ..."
FASTTEST=/data/utils/fasttext

fast_model=${preprocessed}.shuf
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
${FASTTEST} skipgram \
	-input ${preprocessed} \
	-output ${fast_model} \
	-lr ${lr} \
  	-dim ${dim} \
	-ws ${ws} \
	-epoch ${epoch} \
	-minCount ${minCount} \
	-neg ${neg} \
	-loss ns \
	-bucket ${bucket} \
  	-minn ${minn} \
	-maxn ${maxn} \
	-thread ${thread} \
	-t 1e-4 \
	-lrUpdateRate 100 >log/fasttext.log.${ts} 2>&1

echo "generate fasttext dict ..."
awk 'NR>2{print $1}' ${fast_model}.vec > ${fast_model}.dict

echo "filter video dict ..."
python filter_dict.py \
        --input_fasttext_dict_file ${fast_model}.dict \
        --input_video_dict_file ${preprocessed}.video_dict \
        --output_fasttext_subset_dict_file ${fast_model}.subset

echo "fasttext nnsubset ..."
nn_k=100
${FASTTEST} multi-nnsubset ${fast_model}.bin ${fast_model}.dict ${fast_model}.subset ${parallel} ${nn_k}
mv ${fast_model}.dict.result ${fast_model}.result.raw

rm -rf ${final_data_dir}.bak
if [ -d ${final_data_dir} ]; then
    mv ${final_data_dir} ${final_data_dir}.bak
fi
mv ${data_dir} ${final_data_dir}
