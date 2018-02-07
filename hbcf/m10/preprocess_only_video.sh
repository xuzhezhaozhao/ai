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
/data/preprocess/build/src/preprocess \
        -raw_input=${sorted_file} \
        -with_header=false \
        -only_video=true \
        -interval=1000000 \
        -output_user_watched_file=${preprocessed} \
        -output_user_watched_ratio_file=${preprocessed}.watched_ratio \
        -output_video_play_ratio_file=${preprocessed}.play_raito \
        -user_min_watched=${user_min_watched} \
        -user_max_watched=${user_max_watched} \
        -user_abnormal_watched_thr=${user_abnormal_watched_thr} \
        -supress_hot_arg1=${supress_hot_arg1} \
        -supress_hot_arg2=${supress_hot_arg2} \
        -user_effective_watched_time_thr=${user_effective_watched_time_thr} \
        -user_effective_watched_ratio_thr=${user_effective_watched_ratio_thr} \
        -min_count=${min_count} \
        -ban_algo_watched_ratio_thr=${ban_algo_watched_ratio_thr} \
        -ban_algo_ids=${ban_algo_ids} \
        -video_play_ratio_bias=${video_play_ratio_bias} \
        -output_video_dict_file=${preprocessed}.video_dict \
        -output_article_dict_file=${preprocessed}.article_dict \
        -output_video_click_file=${preprocessed}.video_click


echo "fastText train ..."
fast_model=${preprocessed}.shuf
minCount=${min_count}
lr=0.025
dim=100
ws=15
epoch=3
neg=5
bucket=10
minn=0
maxn=0
thread=${parallel}
ts=`date +%Y%m%d%H%M%S`
/data/utils/fasttext skipgram \
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

# call newman's disable interface
# /data/utils/enable_filter -input ${fast_model}.dict -output_valid ${fast_model}.dict.valid -output_filtered ${fast_model}.dict.filtered

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
