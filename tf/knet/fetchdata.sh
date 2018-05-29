#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

hadoop_bin=/usr/local/services/hadoop_client_2_2_0-1.0/tdwdfsclient/bin/hadoop
hdfs_data_path=hdfs://ss-sng-dc-v2/stage/outface/SNG/g_sng_im_sng_imappdev_tribe/zhezhaoxu/preprocessed_data/hbcf

mkdir -p tmp_hdfs
${hadoop_bin} fs \
    -Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017,g_sng_im_sng_imappdev_tribe \
    -get ${hdfs_data_path} tmp_hdfs

cat tmp_hdfs/hbcf/part* > ./data.in
rm -rf tmp_hdfs/

echo "shuf ..."
shuf data.in -o data.in.shuf
total_lines=$(wc -l data.in.shuf | awk '{print $1}')
eval_lines=500000
train_lines=$((total_lines-eval_lines))

echo "generate train_data ..."
head data.in.shuf -n ${train_lines} > train_data.in

echo "generate eval_data ..."
tail data.in.shuf -n ${eval_lines} > eval_data.in
