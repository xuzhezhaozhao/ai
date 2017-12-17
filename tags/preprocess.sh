#! /usr/bin/env bash

ft_in=data/tags.ft.in

python tags.py --input_video_tags_file data/video_tags.csv --input_tag_info_file data/taginfo.csv --input_article_tags_file data/article_tags.csv --output ${ft_in}
