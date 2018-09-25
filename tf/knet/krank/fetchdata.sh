#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

# fetch video and article history

parallel=47

raw_data_dir=raw_data
tmp_hdfs_dir=tmp_hdfs
hadoop_bin=/usr/local/services/hadoop_client_2_2_0-1.0/tdwdfsclient/bin/hadoop
hdfs_data_path=hdfs://ss-sng-dc-v2/stage/outface/SNG/g_sng_im_sng_imappdev_tribe/zhezhaoxu/preprocessed_data/hbcf2_V2
rowkey_count_path=hdfs://ss-sng-dc-v2/stage/outface/SNG/g_sng_im_sng_imappdev_tribe/zhezhaoxu/preprocessed_data/rowkey_count_V2

base_hdfs_data_path=`basename ${hdfs_data_path}`
base_rowkey_count_path=`basename ${rowkey_count_path}`

echo "begin fetch video and article watch data ..."
echo "check data timestamp .."
array_check=($hdfs_file_path $rowkey_count_path)
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


parallel_download() {
${hadoop_bin} fs \
    -Dhadoop.job.ugi=tdw_alwensong:tdw2018,g_sng_qqkd_sng_buluo_kd_video_group \
    -ls $1 > ${tmp_hdfs_dir}/ls_cmd
base_dir=`basename $1`
mkdir -p ${tmp_hdfs_dir}/${base_dir}
echo "echo ls"
sed 's/.*hdfs:/hdfs:/g' tmp_hdfs/ls_cmd > tmp_hdfs/file_list
echo "satrt get"
for line in `cat ./tmp_hdfs/file_list`
do
	if [[ $line =~ "csv" ]]
	then
		nohup ${hadoop_bin} fs \
            -Dhadoop.job.ugi=tdw_alwensong:tdw2018,g_sng_qqkd_sng_buluo_kd_video_group \
            -get $line ${tmp_hdfs_dir}/${base_dir} &
	fi
done
wait
}

rm -rf ${tmp_hdfs_dir}/${base_hdfs_data_path}
rm -rf ${tmp_hdfs_dir}/${base_rowkey_count_path}
parallel_download ${hdfs_data_path}
parallel_download ${rowkey_count_path}

let sz=`du -s -BG ${tmp_hdfs_dir}/${base_hdfs_data_path} | awk -F 'G' '{print $1}'`
if [ $sz -le 10 ]; then
   echo "fetch hdfs may be failed, size = ${sz}G"
   exit 1
fi

mkdir -p ${raw_data_dir}
cat ${tmp_hdfs_dir}/${base_hdfs_data_path}/part* > ${raw_data_dir}/data.in
cat ${tmp_hdfs_dir}/${base_rowkey_count_path}/part* > ${raw_data_dir}/rowkey_count.csv
rm -rf ${tmp_hdfs_dir}/${base_hdfs_data_path}
rm -rf ${tmp_hdfs_dir}/${base_rowkey_count_path}
echo "fetch video and article watch data done."

total_lines=$(wc -l ${raw_data_dir}/data.in | awk '{print $1}')
eval_lines=100000
train_lines=$((total_lines-eval_lines))

echo "generate train_data ..."
head ${raw_data_dir}/data.in -n ${train_lines} > ${raw_data_dir}/train_data.in

echo "generate eval_data ..."
tail ${raw_data_dir}/data.in -n ${eval_lines} > ${raw_data_dir}/eval_data.in
rm -rf ${raw_data_dir}/data.in
