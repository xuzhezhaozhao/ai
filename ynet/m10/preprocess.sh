#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}


parallel=47

rm -rf data.bak
if [ -d data ]; then
  echo "backup data/ ..."
  mv data data.bak
fi
mkdir -p data

rm -rf records
/data/hadoop_client/new/tdwdfsclient/bin/hadoop fs -Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017,g_sng_im_sng_imappdev_tribe -get hdfs://ss-sng-dc-v2/stage/outface/SNG/g_sng_im_sng_imappdev_tribe/zhezhaoxu/DNN/records

input=data/data.in
sorted_file=${input}.sorted
cat records/part* > ${sorted_file}


echo "transform sorted file to fastText format ..."
preprocessed=${input}.preprocessed
min_count=50
./preprocess/build/src/preprocess \
    -raw_input=${sorted_file} \
    -with_header=false \
    -only_video=true \
    -interval=1000000 \
    -output_user_watched_file=${preprocessed} \
    -output_user_watched_ratio_file=${preprocessed}.watched_ratio \
    -user_min_watched=10 \
    -user_max_watched=1024 \
    -user_abnormal_watched_thr=2048 \
    -supress_hot_arg1=8 \
    -supress_hot_arg2=3 \
    -user_effective_watched_time_thr=10 \
    -user_effective_watched_ratio_thr=0.25 \
    -min_count=${min_count} \
    -ban_algo_watched_ratio_thr=1.1 \
    -ban_algo_ids='3323,3321,3313,3312,3311,3310,3309,3308,3307,3306,3305,3304,3303,3302,3301'


echo "fastText train ..."
# fasttext args
minCount=${min_count}
dim=100
ws=20
epoch=5
neg=5
thread=${parallel}
fast_output=${input}
utils/fasttext \
    skipgram \
    -input ${preprocessed} \
    -output ${fast_output} \
    -lr 0.025 \
    -dim ${dim} \
    -ws ${ws} \
    -epoch ${epoch} \
    -minCount ${minCount} \
    -neg ${neg} \
    -loss ns \
    -bucket 2000000 \
    -minn 0 \
    -maxn 0 \
    -thread ${thread} \
    -t 1e-4 \
    -lrUpdateRate 100

tf_input=${input}.tf
python utils/vec2binary.py \
    --input ${fast_output}.vec \
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
    --max_per_user ${max_per_user}
