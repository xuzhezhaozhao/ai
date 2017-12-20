#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

input=$1

rawdata_dir=raw_data
data_dir=data

output=${data_dir}/$2

echo 'uniq ...'
sort -n ${rawdata_dir}/article_tags.csv | uniq > ${data_dir}/article_tags.csv.uniq
sort -n ${rawdata_dir}/video_tags.csv | uniq > ${data_dir}/video_tags.csv.uniq
sort -n ${rawdata_dir}/taginfo.csv | uniq > ${data_dir}/taginfo.csv.uniq
sort -n ${input} | uniq > ${output}.uniq

echo 'delete csv file header ...'
sed "1d" ${output}.uniq > ${output}.noheader

echo "sort csv file with 1st field ..."
sorted_file=${output}.sorted
mkdir -p tmp_sort/
sort -T tmp_sort/ -t ',' -k 1 --parallel=4 ${output}.noheader -o ${sorted_file}
rm -rf tmp_sort/

output_tags=${data_dir}/record_tags.in

python records.py --input ${output}.sorted --input_article_tags_file ${data_dir}/article_tags.csv.uniq --input_video_tags_file ${data_dir}/video_tags.csv.uniq --output_history_tags ${output_tags} --sort_tags true --input_tag_info_file ${data_dir}/taginfo.csv.uniq --max_lines 20000000

echo 'shuf ...'
shuf -o ${output_tags}.shuf ${output_tags}
