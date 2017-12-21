#! /usr/bin/env bash

ft_in=data/labels.in

rawdata_dir=raw_data
data_dir=data

# --input_article_tags_file ${rawdata_dir}/article_tags.csv \

python labels.py \
    --input_video_tags_file ${rawdata_dir}/video_tags.csv \
    --input_tag_info_file ${rawdata_dir}/taginfo.csv \
    --min_labels 1 \
    --sort_tags true \
    --output_info ${ft_in} \
    --output_label_dict_file ${data_dir}/labelinfo.csv

shuf -o ${ft_in}.shuf ${ft_in}


# fasttext
minCount=1
minn=0
maxn=0
thread=4
dim=100
ws=5
epoch=5
neg=5
lr=0.025

input=${ft_in}.shuf

utils/fasttext supervised \
    -input ${input} \
    -output ${input} \
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
    && awk 'NR>2{print $1}' ${input}.vec > ${input}.dict