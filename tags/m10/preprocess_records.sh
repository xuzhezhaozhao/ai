#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}


rawdata_dir=raw_data
data_dir=data

input=${rawdata_dir}/records.csv

output=${data_dir}/`basename ${input}`

if [ ! -f ${output}.sorted ]; then
    echo "sort csv file with 1st field ..."
    sorted_file=${output}.sorted
    mkdir -p tmp_sort/
    sort -T tmp_sort/ -t ',' -k 1 --parallel=44 ${input} -o ${sorted_file}
    rm -rf tmp_sort/
fi

output_tags=${data_dir}/records_tags.in
python records.py \
    --input ${output}.sorted \
    --input_article_tags_file ${rawdata_dir}/article_tags.csv \
    --input_video_tags_file ${rawdata_dir}/video_tags.csv \
    --input_tag_info_file ${rawdata_dir}/taginfo.csv \
    --output_history_tags ${output_tags} \
    --sort_tags false \
    --min_items 5 \
    --max_items 1024 \
    --max_lines -1

echo 'shuf ...'
shuf -o ${output_tags}.shuf ${output_tags}


# fasttext
ft_in=${output_tags}.shuf
minCount=300
minn=0
maxn=0
thread=4
dim=100
ws=20
epoch=5
neg=5
lr=0.025

utils/fasttext skipgram \
    -input ${ft_in} \
    -output ${ft_in} \
    -lr ${lr} \
    -dim ${dim} \
    -ws ${ws} \
    -epoch ${epoch} \
    -minCount ${minCount} \
    -neg ${neg} \
    -loss ns \
    -bucket 2000000 \
    -minn ${minn} \
    -maxn ${maxn} \
    -thread ${thread} \
    -t 1e-4 \
    -lrUpdateRate 100  \
    && awk 'NR>2{print $1}' ${ft_in}.vec > ${ft_in}.dict
