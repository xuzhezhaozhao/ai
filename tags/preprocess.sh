#! /usr/bin/env bash

ft_in=data/tags.in
ft_in_raw=data/tags.raw.in

python tags.py --input_video_tags_file data/video_tags.csv --input_tag_info_file data/taginfo.csv --input_article_tags_file data/article_tags.csv --output_info ${ft_in} --output_raw ${ft_in_raw}

shuf -o ${ft_in}.shuf ${ft_in}
shuf -o ${ft_in_raw}.shuf ${ft_in_raw}
