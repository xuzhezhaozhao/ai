#! /usr/bin/env bash

rm -rf data.bak
mv data data.bak

/data/hadoop_client/new/tdwdfsclient/bin/hadoop fs -Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017,g_sng_im_sng_imappdev_tribe -get hdfs://ss-sng-dc-v2/stage/outface/SNG/g_sng_im_sng_imappdev_tribe/zhezhaoxu/similar_videos_csv/data .

if [$? -eq 0]; then
  echo 'hdfs failed'
  exit
fi

cat data/* > data.all

input=data.all
output=${input}.fasttext

# max items per user
klimit=1024
# fasttext min count
minCount=50

sorted_file=${input}.sorted

echo "sort input file ..."
mkdir -p tmp/
sort -T tmp/ -t ',' -k 4 --parallel=24 ${input} -o ${sorted_file}

echo "transform sorted file ..."
./transform.py ${sorted_file} ${output} ${klimit}

echo "fastText train ..."
fast_model=${output}.model
~/fastText/fasttext skipgram -input ${output} -output ${fast_model} -lr 0.025\
  -dim 100 -ws 5 -epoch 5 -minCount ${minCount} -neg 5 -loss ns -bucket 2000000\
  -minn 0 -maxn 0 -thread 4 -t 1e-4 -lrUpdateRate 100

echo "generate query list ..."
awk '{print $1}' ${fast_model}.vec > ${fast_model}.query
echo "remove first two lines of query list ..."
sed -i "1d" ${fast_model}.query
sed -i "1d" ${fast_model}.query

echo "split query list ..."
split -d -n l/3 ${fast_model}.query ${fast_model}.query.

echo "fastText nn ..."
FASTTEST=~/fastText/fasttext
${FASTTEST} nn ${fast_model}.bin 50 30 < ${fast_model}.query.00 > ${fast_model}.result.00 &
${FASTTEST} nn ${fast_model}.bin 50 30 < ${fast_model}.query.01 > ${fast_model}.result.01 &
${FASTTEST} nn ${fast_model}.bin 50 30 < ${fast_model}.query.02 > ${fast_model}.result.02 &
wait
wait
wait
cat ${fast_model}.result.00 ${fast_model}.result.01 ${fast_model}.result.02 > ${fast_model}.result
