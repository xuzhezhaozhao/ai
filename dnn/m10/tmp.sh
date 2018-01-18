#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

parallel=47

input=raw_data/data.in
sorted_file=${input}.sorted

ynet_data_dir=ynet/data_ing
final_data_dir=ynet/data
preprocessed=${ynet_data_dir}/data.in.preprocessed
mkdir -p ${ynet_data_dir}


echo "transform sorted file to fastText format ..."
min_count=10
user_min_watched=10
user_max_watched=1024
user_abnormal_watched_thr=2048
user_effective_watched_time_thr=0
user_effective_watched_ratio_thr=0.1
ban_algo_ids='3323,3321,3313,3312,3311,3310,3309,3308,3307,3306,3305,3304,3303,3302,3301'
ban_algo_watched_ratio_thr=1.8
video_play_ratio_bias=30
supress_hot_arg1=-1
supress_hot_arg2=3

/data/preprocess/build/src/preprocess \
    -raw_input=${sorted_file} \
    -with_header=false \
    -only_video=true \
    -interval=1000000 \
    -output_user_watched_file=${preprocessed} \
    -output_user_watched_ratio_file=${preprocessed}.watched_ratio \
    -output_video_play_ratio_file=${preprocessed}.play_ratio \
    -user_min_watched=${user_min_watched} \
    -user_max_watched=${user_max_watched} \
    -user_abnormal_watched_thr=${user_abnormal_watched_thr} \
    -supress_hot_arg1=${supress_hot_arg1} \
    -supress_hot_arg2=3 \
    -user_effective_watched_time_thr=${user_effective_watched_time_thr} \
    -user_effective_watched_ratio_thr=${user_effective_watched_ratio_thr} \
    -min_count=${min_count} \
    -ban_algo_watched_ratio_thr=${ban_algo_watched_ratio_thr} \
    -ban_algo_ids=${ban_algo_ids} \
    -video_play_ratio_bias=${video_play_ratio_bias} \
    -output_video_dict_file=${preprocessed}.video_dict \
    -output_article_dict_file=${preprocessed}.article_dict \
    -output_video_click_file=${preprocessed}.video_click

mkdir -p log
echo "fastText train ..."
fast_model=${ynet_data_dir}/data.in
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
/data/utils/fasttext \
    skipgram \
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
    -lrUpdateRate 100 >log/fasttext.ynet.log.${ts} 2>&1

tf_input=${ynet_data_dir}/data.in.tf
python utils/vec2binary.py \
    --input ${fast_model}.vec \
    --output ${tf_input}.vec \
    --output_dict_file ${tf_input}.dict

max_per_user=100
watched_size=20
watched_size_pctr=21
python utils/records2binary.py \
    --input_records ${preprocessed} \
    --input_dict_file ${tf_input}.dict \
    --input_watched_ratio_file ${preprocessed}.watched_ratio \
    --output_watched ${tf_input}.watched \
    --output_watched_pctr ${tf_input}.watched.pctr \
    --output_predicts ${tf_input}.predicts \
    --output_predicts_pctr ${tf_input}.predicts.pctr \
    --watched_size ${watched_size} \
    --watched_size_pctr ${watched_size_pctr} \
    --max_per_user ${max_per_user} \
    --calculate_recall_inputs 0 

python utils/pctr_transform.py \
    --input_records ${preprocessed} \
    --input_watched_ratio_file ${preprocessed}.watched_ratio \
    --output_watched_pctr ${preprocessed}.pctr.out \
    --watched_size_pctr 10 \
    --class_num_pctr 10 \
    --pctr_step 2
shuf -o ${preprocessed}.pctr.out.shuf ${preprocessed}.pctr.out
/data/utils/fasttext supervised -input ${preprocessed}.pctr.out.shuf -output ${fast_model}.pctr -minCount 10 -minn 0 -maxn 0 -lr 0.05 -ws 10 -epoch 5 -thread 44


rm -rf ${final_data_dir}.bak
if [ -d ${final_data_dir} ]; then
    mv ${final_data_dir} ${final_data_dir}.bak
fi
mv ${ynet_data_dir} ${final_data_dir}
