#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

rawdata_dir=raw_data
data_dir=data
fetch_dir=kandian_tag_extent

rm -rf ${data_dir}.bak
rm -rf ${rawdata_dir}.bak
rm -rf ${fetch_dir}.bak
if [ -d ${rawdata_dir} ]; then
  echo "backup ${rawdata_dir} ..."
  mv ${rawdata_dir} ${rawdata_dir}.bak
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
mkdir -p ${rawdata_dir}
mkdir -p ${data_dir}
cat ${fetch_dir}/article_info/part* > ${rawdata_dir}/article_tags.csv
cat ${fetch_dir}/video_info/part* > ${rawdata_dir}/video_tags.csv
cat ${fetch_dir}/tag_info/part* > ${rawdata_dir}/taginfo.csv
cat ${fetch_dir}/records/part* > ${rawdata_dir}/records.csv

cat ${fetch_dir}/class1_info/part* > ${rawdata_dir}/only_article_class1.info
sed -i 's/$/ 1/' ${rawdata_dir}/only_article_class1.info
cat ${fetch_dir}/class2_info/part* > ${rawdata_dir}/only_article_class2.info
sed -i 's/$/ 2/' ${rawdata_dir}/only_article_class2.info

echo "fetch soso tag dict from hdfs ..."
sosodictaddr=`/data/hadoop_client/new/tdwdfsclient/bin/hadoop fs -Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017 -ls  hdfs://tl-if-nn-tdw.tencent-distribute.com:54310/stage/outface/TEG/g_teg_ainlp_ainlp/tag/tagid/tag_dict.* 2>/dev/null | awk '{print $8}' | sort | tail -n1`
sosodictname=`basename ${sosodictaddr}`
/data/hadoop_client/new/tdwdfsclient/bin/hadoop fs -Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017,g_sng_im_sng_imappdev_tribe -get ${sosodictaddr} ${rawdata_dir}/
# mv ${rawdata_dir}/${sosodictname} ${rawdata_dir}/soso_tagdict.csv
iconv -f gb18030 -t utf-8 ${rawdata_dir}/${sosodictname} > ${data_dir}/soso_tagdict.csv.utf8


echo "train classifier ..."
./preprocess_records.sh
./preprocess_classifier.sh
./preprocess_classifier_only_article.sh
# ./preprocess_classifier_with_article.sh

echo "send udp ..."
./utils/sendupdate -modid=900481 -cmdid=65536
