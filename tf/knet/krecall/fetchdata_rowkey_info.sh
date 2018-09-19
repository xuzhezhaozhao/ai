#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

hdfs_dir=tmp_hdfs/rowkey_info
rawdata_dir=raw_data
parallel=47
hadoop_bin=/usr/local/services/hadoop_client_2_2_0-1.0/tdwdfsclient/bin/hadoop

echo "check data timestampe ..."
hdfs_rowkey_info=`${hadoop_bin} fs -Dhadoop.job.ugi=tdw_alwensong:tdw2018,g_sng_qqkd_sng_buluo_kd_video_group -ls hdfs://ss-sng-dc-v2/data/SPARK/SNG/g_sng_qqkd_sng_buluo_kd_video_group/frogli/video_info/all_inuse_video.*.done 2>/dev/null | awk '{print $8}' | sort | tail -n1`
hdfs_file_path=${hdfs_rowkey_info}
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

echo "fetch data from hdfs ..."
rm -rf ${hdfs_dir}
mkdir -p ${hdfs_dir}
rowkey_info=`basename ${hdfs_rowkey_info}`
echo "fetch rowkey_info file " ${rowkey_info} " into " ${hdfs_dir}
${hadoop_bin} fs -Dhadoop.job.ugi=tdw_zhezhaoxu:zhao2017,g_sng_im_sng_imappdev_tribe -get ${hdfs_rowkey_info} ${hdfs_dir}/

# check data
rowkey_info_lines=`wc -l ${hdfs_dir}/${rowkey_info} | awk '{print $1}'`
if [ ${rowkey_info_lines} -le 10000 ]; then
    echo "fetch rowkey_info may be failed, lines = ${rowkey_info_lines}"
    exit 1
fi

cp ${hdfs_dir}/${rowkey_info} ${rawdata_dir}/rowkey_info.json
