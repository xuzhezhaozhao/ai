#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

raw_data_dir=tmp_hdfs/data1
data_dir=data1_ing
final_data_dir=data1

parallel=47

echo "fetch data from hdfs ..."
rm -rf tmp_hdfs
mkdir -p tmp_hdfs
/data/hadoop_client/new/tdwdfsclient/bin/hadoop fs -Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017,g_sng_im_sng_imappdev_tribe -get hdfs://ss-sng-dc-v2/stage/outface/SNG/g_sng_im_sng_imappdev_tribe/zhezhaoxu/kandian_similar_videos_csv/data1 tmp_hdfs

let sz=`du -sh tmp_hdfs/ | awk -F 'G' '{print $1}'`
if [ $sz -le 30 ]; then
   echo "fetch hdfs may be failed, size = ${sz}G"
   exit 1
fi

mkdir -p ${data_dir}
input=${data_dir}/data.in
cat ${raw_data_dir}/part* > ${input}

sorted_file=${input}.sorted
echo "sort input file ..."
mkdir -p tmp_sort/
sort -T tmp_sort/ -t ',' -k 1 --parallel=${parallel} ${input} -o ${sorted_file}
rm -rf tmp_sort/

echo "transform sorted file ..."
user_min_watched=20
user_max_watched=512
user_abnormal_watched_thr=2048
user_effective_watched_time_thr=20
user_effective_watched_ratio_thr=0.8

preprocessed=${input}.preprocessed
./preprocess/build/src/preprocess \
	-raw_input=${sorted_file} \
	-with_header=false \
	-only_video=true \
	-interval=1000000 \
	-output_user_watched_file=${preprocessed} \
	-user_min_watched=${user_min_watched} \
	-user_max_watched=${user_max_watched} \
	-user_abnormal_watched_thr=${user_abnormal_watched_thr} \
	-supress_hot_arg1=8 \
	-supress_hot_arg2=3 \
        -user_effective_watched_time_thr=${user_effective_watched_time_thr} \
        -user_effective_watched_ratio_thr=${user_effective_watched_ratio_thr}

shuf -o ${preprocessed}.shuf ${preprocessed}

echo "fastText train ..."
fast_model=${preprocessed}.shuf
minCount=50
lr=0.025
dim=100
ws=15
epoch=5
neg=5
bucket=2000000
minn=0
maxn=0
thread=${parallel}
ts=`date '+%s'`
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
	-lrUpdateRate 100 >fasttext.log.${ts} 2>&1

echo "generate fasttext dict ..."
awk 'NR>2{print $1}' ${fast_model}.vec > ${fast_model}.dict


rm -rf ${fast_model}.query.*
echo "split query list ..."
split -d -n l/${parallel} ${fast_model}.dict ${fast_model}.query.

echo "fasttext nn ..."
nn_cnt=100
FASTTEST=./utils/fasttext

rm -rf ${fast_model}.query.*.result
rm -rf ${fast_model}.result
queryfiles=`ls ${fast_model}.query.*`
for queryfile in ${queryfiles[@]}
do
    echo ${queryfile}
    ${FASTTEST} nn ${fast_model}.bin ${nn_cnt} < ${queryfile} > ${queryfile}.result &
done

for queryfile in ${queryfiles[@]}
do
    echo "${queryfile} wait ..."
    wait
done

cat ${fast_model}.query.*.result > ${fast_model}.result
rm -rf ${fast_model}.query.*

rm -rf ${final_data_dir}.bak
if [ -d ${final_data_dir} ]; then
    mv ${final_data_dir} ${final_data_dir}.bak
fi
mv ${data_dir} ${final_data_dir}

./utils/sendupdate -modid=907457 -cmdid=65536
