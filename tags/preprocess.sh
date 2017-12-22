#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

raw_data_dir=raw_data
data_dir=data
fetch_dir=kandian_tag_extent

rm -rf ${data_dir}.bak
rm -rf ${raw_data_dir}.bak
rm -rf ${fetch_dir}.bak
if [ -d ${raw_data_dir} ]; then
  echo "backup ${raw_data_dir} ..."
  mv ${raw_data_dir} ${raw_data_dir}.bak
fi
if [ -d ${data_dir} ]; then
  echo "backup ${data_dir} ..."
  mv ${data_dir} ${data_dir}.bak
fi
if [ -d ${fetch_dir} ]; then
  echo "backup ${fetch_dir} ..."
  mv ${fetch_dir} ${fetch_dir}.bak
fi


echo "fetch kandian_tag_extent from hdfs ..."
/data/hadoop_client/new/tdwdfsclient/bin/hadoop fs -Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017,g_sng_im_sng_imappdev_tribe -get hdfs://ss-sng-dc-v2/stage/outface/SNG/g_sng_im_sng_imappdev_tribe/zhezhaoxu/kandian_tag_extent/ .
mkdir -p ${raw_data_dir}
mkdir -p ${data_dir}
cat ${fetch_dir}/article_info/part* > ${raw_data_dir}/article_tags.csv
cat ${fetch_dir}/video_info/part* > ${raw_data_dir}/video_tags.csv
cat ${fetch_dir}/tag_info/part* > ${raw_data_dir}/taginfo.csv


echo "fetch soso tag dict from hdfs ..."
sosodictaddr=`/data/hadoop_client/new/tdwdfsclient/bin/hadoop fs -Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017 -ls  hdfs://tl-if-nn-tdw.tencent-distribute.com:54310/stage/outface/TEG/g_teg_ainlp_ainlp/tag/tagid/tag_dict.* 2>/dev/null | awk '{print $8}' | sort | tail -n1`
sosodictname=`basename ${sosodictaddr}`
/data/hadoop_client/new/tdwdfsclient/bin/hadoop fs -Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017,g_sng_im_sng_imappdev_tribe -get ${sosodictaddr} ${raw_data_dir}/
mv ${raw_data_dir}/${sosodictname} ${raw_data_dir}/soso_tagdict.csv
