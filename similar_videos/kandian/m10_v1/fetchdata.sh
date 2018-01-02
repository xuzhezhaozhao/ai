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
sorted_file=${input}.sorted
cat ${raw_data_dir}/part* > ${sorted_file}
