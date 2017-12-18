#! /usr/bin/env bash

ft_in=data/tags.in
ft_in_raw=data/tags.raw.in

sort -n data/article_tags.csv | uniq > data/article_tags.csv.uniq
sort -n data/video_tags.csv | uniq > data/video_tags.csv.uniq
sort -n data/taginfo.csv | uniq > data/taginfo.csv.uniq

python tags.py --input_video_tags_file data/video_tags.uniq.csv.uniq --input_tag_info_file data/taginfo.csv.uniq --input_article_tags_file data/article_tags.csv.uniq --output_info ${ft_in} --output_raw ${ft_in_raw} --min_labels 2 --sort_tags true

shuf -o ${ft_in}.shuf ${ft_in}
shuf -o ${ft_in_raw}.shuf ${ft_in_raw}
