#! /usr/bin/env bash

ft_in=data/tags.in
rawdata_dir=raw_data

python tags.py \
    --input_video_tags_file ${rawdata_dir}/video_tags.csv \
    --input_tag_info_file ${rawdata_dir}/taginfo.csv \
    --input_article_tags_file ${rawdata_dir}/article_tags.csv \
    --min_labels 2 \
    --sort_tags true \
    --output_info ${ft_in}

shuf -o ${ft_in}.shuf ${ft_in}
