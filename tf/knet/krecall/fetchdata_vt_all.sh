#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

# fetch video and article history

parallel=47

raw_data_dir=raw_data
tmp_hdfs_dir=tmp_hdfs
hadoop_bin=/usr/local/services/hadoop_client_2_2_0-1.0/tdwdfsclient/bin/hadoop
hdfs_data_path=hdfs://ss-sng-dc-v2/stage/outface/SNG/g_sng_im_sng_imappdev_tribe/zhezhaoxu/preprocessed_data/hbcf2
base_hdfs_data_path=`basename ${hdfs_data_path}`

echo "begin fetch video and article watch data ..."
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

let sz=`du -sh ${tmp_hdfs_dir}/${base_hdfs_data_path} | awk -F 'G' '{print $1}'`
if [ $sz -le 10 ]; then
   echo "fetch hdfs may be failed, size = ${sz}G"
   exit 1
fi

cat ${tmp_hdfs_dir}/${base_hdfs_data_path}/part* > ${raw_data_dir}/data.vt.in
rm -rf ${tmp_hdfs_dir}/${base_hdfs_data_path}

echo "shuf ..."
shuf ${raw_data_dir}/data.vt.in -o ${raw_data_dir}/data.vt.in.shuf

total_lines=$(wc -l ${raw_data_dir}/data.vt.in.shuf | awk '{print $1}')
eval_lines=10000
train_lines=$((total_lines-eval_lines))

echo "generate train_data ..."
head ${raw_data_dir}/data.vt.in.shuf -n ${train_lines} > ${raw_data_dir}/train_data.vt.in

num_workers=2
# split -a 2 -d -n l/${num_workers} ${raw_data_dir}/train_data.vt.in ${raw_data_dir}/train_data.vt.in.

echo "generate eval_data ..."
tail ${raw_data_dir}/data.vt.in.shuf -n ${eval_lines} > ${raw_data_dir}/eval_data.vt.in

echo "fetch video and article watch data done."
