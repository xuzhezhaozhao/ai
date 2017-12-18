#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

input=$1

echo 'delete csv file header ...'
#sed -n '1p' ${input} > ${input}.header
#sed -i "1d" ${input}

echo "sort csv file with 1st field ..."
sorted_file=${input}.sorted
#mkdir -p tmp_sort/
#sort -T tmp_sort/ -t ',' -k 1 --parallel=4 ${input} -o ${sorted_file}
#rm -rf tmp_sort/


output_tags=records_tags.in
output_raw=records_raw.in

python records.py --input ${input}.sorted --input_article_tags_file data/article_tags.csv --input_video_tags_file data/video_tags.csv --output_history_raw ${output_raw} --output_history_tags ${output_tags} --sort_tags true --input_tag_info_file data/taginfo.csv

shuf -o ${output_tags}.shuf ${output_tags}
shuf -o ${output_raw}.shuf ${output_raw}
