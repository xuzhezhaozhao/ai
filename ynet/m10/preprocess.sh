#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

input=data/data.in

rm -rf data.bak
if [ -d data ]; then
  echo "backup data/ ..."
  mv data data.bak
fi
mkdir -p data

/data/hadoop_client/new/tdwdfsclient/bin/hadoop fs -Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017,g_sng_im_sng_imappdev_tribe -get hdfs://ss-sng-dc-v2/stage/outface/SNG/g_sng_im_sng_imappdev_tribe/zhezhaoxu/DNN/records

cat records/part* > data/data.in

echo "sort csv file with 1st field ..."
sorted_file=${input}.sorted
mkdir -p tmp_sort/
sort -T tmp_sort/ -t ',' -k 1 --parallel=4 ${input} -o ${sorted_file}
rm -rf tmp_sort/

preprocessed=${input}.preprocessed
echo "transform sorted file to fastText format ..."
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
    -user_effective_watched_time_thr=20 \
    -user_effective_watched_ratio_thr=0.3 \
    -min_count=${min_count}

echo "fastText train ..."
# fasttext args
minCount=${min_count}
dim=100
ws=20
epoch=5
neg=5
thread=4
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

#python utils/filter_transform.py \
    #--input ${preprocessed} \
    #--output ${tf_input}.filterd \
    #--dictfile ${tf_input}.dict

max_per_user=100
watched_size=5
watched_size_pctr=20
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
