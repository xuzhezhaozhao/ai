#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

raw_data_dir=raw_data1
data_dir=data1

rm -rf ${data_dir}.bak
rm -rf ${raw_data_dir}.bak
if [ -d ${raw_data_dir} ]; then
  echo "backup ${raw_data_dir} ..."
  mv ${raw_data_dir} ${raw_data_dir}.bak
fi
if [ -d ${data_dir} ]; then
  echo "backup ${data_dir} ..."
  mv ${data_dir} ${data_dir}.bak
fi


echo "fetch data from hdfs ..."
/data/hadoop_client/new/tdwdfsclient/bin/hadoop fs -Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017,g_sng_im_sng_imappdev_tribe -get hdfs://ss-sng-dc-v2/stage/outface/SNG/g_sng_im_sng_imappdev_tribe/zhezhaoxu/kandian_similar_videos_csv/data1 .

mv data1 ${raw_data_dir}
mkdir -p ${data_dir}

input=${data_dir}/data.in
cat ${raw_data_dir}/part* > ${input}


sorted_file=${input}.sorted
echo "sort input file ..."
mkdir -p tmp_sort/
sort -T tmp_sort/ -t ',' -k 1 --parallel=44 ${input} -o ${sorted_file}
rm -rf tmp_sort/

echo "transform sorted file ..."
user_min_watched=20
user_max_watched=512
user_abnormal_watched_thr=2048
preprocessed=${input}.preprocessed
./preprocess/build/src/preprocess \
	-raw_input=${sorted_file} \
	-with_header=false \
	-only_video=true \
	-interval=1000000 \
	-output_user_watched_file=${preprocessed} \
	-user_min_watched=${user_min_watched} \
	-user_max_watched=${user_max_watched} \
	-user_abnormal_watched_thr=${user_abnormal_watched_thr}

shuf -o ${preprocessed}.shuf ${preprocessed}

echo "fastText train ..."
fast_model=${preprocessed}.shuf
minCount=50
lr=0.025
dim=100
ws=40
epoch=5
neg=5
bucket=2000000
minn=0
maxn=0
thread=44
./utils/fasttext skipgram \
	-input ${preprocessed}.shuf \
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
	-lrUpdateRate 100

echo "generate fasttext dict ..."
awk 'NR>2{print $1}' ${fast_model}.vec > ${fast_model}.dict
