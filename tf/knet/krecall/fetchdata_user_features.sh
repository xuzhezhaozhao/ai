#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

# fetch user features

parallel=47

raw_data_dir=raw_data
tmp_hdfs_dir=tmp_hdfs
hadoop_bin=/usr/local/services/hadoop_client_2_2_0-1.0/tdwdfsclient/bin/hadoop
hdfs_data_path=hdfs://ss-sng-dc-v2/stage/outface/SNG/g_sng_im_sng_imappdev_tribe/zhezhaoxu/preprocessed_data/user_features
base_hdfs_data_path=`basename ${hdfs_data_path}`

echo "begin fetch user features ..."
echo "check data timestamp .."
array_check=($hdfs_file_path)
today_timestamp=$(date -d "$(date +"%Y-%m-%d %H:%M")" +%s)
out_of_date_hours=24
checkOutDate() {
    ${hadoop_bin} fs \
        -Dhadoop.job.ugi=tdw_alwensong:tdw2018,g_sng_qqkd_sng_buluo_kd_video_group \
        -ls $1 > temp.txt
    cat temp.txt | while read quanxian temp user group size day hour filepath
    do
        current_file_time="$day $hour"
        current_file_timestamp=$(date -d "$current_file_time" +%s)
        if [ $(($today_timestamp-$current_file_timestamp)) -ge $((${out_of_date_hours}*60*60)) ];then
            echo "$(date +'%Y-%m-%d %H:%M:%S') $1 out of date"
            exit -1
        fi
    done
}

for filename in ${array_check[@]}
do
    echo "$(date +'%Y-%m-%d %H:%M:%S') processing filepath: $filename"
    checkOutDate $filename
    echo -e "\n"
done


rm -rf ${tmp_hdfs_dir}/${base_hdfs_data_path}
${hadoop_bin} fs \
    -Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017,g_sng_im_sng_imappdev_tribe \
    -get ${hdfs_data_path} ${tmp_hdfs_dir}


cat ${tmp_hdfs_dir}/${base_hdfs_data_path}/part* > ${raw_data_dir}/user_features.tsv
rm -rf ${tmp_hdfs_dir}/${base_hdfs_data_path}

total_lines=$(wc -l ${raw_data_dir}/user_features.tsv | awk '{print $1}')
if [ ${total_lines} -le 1000000 ]; then
   echo "fetch user_features hdfs may be failed, total_lines = ${total_lines}"
   exit 1
fi
echo "fetch user features done."
