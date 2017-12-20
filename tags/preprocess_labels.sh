#! /usr/bin/env bash

ft_in=data/labels.in

rawdata_dir=raw_data
data_dir=data

python labels.py \
    --input_video_tags_file ${rawdata_dir}/video_tags.csv \
    --input_tag_info_file ${rawdata_dir}/taginfo.csv \
    --min_labels 1 \
    --sort_tags true \
    --output_info ${ft_in} \
    --output_label_dict_file ${data_dir}/labelinfo.csv

shuf -o ${ft_in}.shuf ${ft_in}
