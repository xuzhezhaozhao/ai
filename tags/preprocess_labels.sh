#! /usr/bin/env bash

ft_in=data/labels.in

rawdata_dir=raw_data
data_dir=data

echo "uniq..."
sort -n ${rawdata_dir}/article_tags.csv | uniq > ${data_dir}/article_tags.csv.uniq
sort -n ${rawdata_dir}/video_tags.csv | uniq > ${data_dir}/video_tags.csv.uniq
sort -n ${rawdata_dir}/taginfo.csv | uniq > ${data_dir}/taginfo.csv.uniq

python labels.py --input_video_tags_file ${data_dir}/video_tags.csv.uniq --input_tag_info_file ${data_dir}/taginfo.csv.uniq --input_article_tags_file ${data_dir}/article_tags.csv.uniq --output_info ${ft_in} --min_labels 2 --sort_tags true --output_label_dict_file ${data_dir}/labels.dict

shuf -o ${ft_in}.shuf ${ft_in}
