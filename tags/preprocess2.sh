#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

input=$1

echo 'uniq ...'
sort -n data/article_tags.csv | uniq > data/article_tags.csv.uniq
sort -n data/video_tags.csv | uniq > data/video_tags.csv.uniq
sort -n data/taginfo.csv | uniq > data/taginfo.csv.uniq
sort -n ${input} | uniq > ${input}.uniq

echo 'delete csv file header ...'
#sed -n '1p' ${input}.uniq > ${input}.header
#sed -i "1d" ${input}.uniq

echo "sort csv file with 1st field ..."
sorted_file=${input}.sorted
mkdir -p tmp_sort/
sort -T tmp_sort/ -t ',' -k 1 --parallel=4 ${input}.uniq -o ${sorted_file}
rm -rf tmp_sort/

output_tags=data/record_tags.in
output_raw=data/record_raw.in

python records.py --input ${input}.sorted --input_article_tags_file data/article_tags.csv.uniq --input_video_tags_file data/video_tags.csv.uniq --output_history_raw ${output_raw} --output_history_tags ${output_tags} --sort_tags true --input_tag_info_file data/taginfo.csv.uniq --max_lines 10000000

shuf -o ${output_tags}.shuf ${output_tags}
shuf -o ${output_raw}.shuf ${output_raw}
