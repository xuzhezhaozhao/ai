
ft_in=data/tags.ft.in

python tags.py --input_video_tags_file data/video_tags.csv --input_tag_info_file data/taginfo.csv --input_article_tags_file data/article_tags.csv --output ${ft_in}

ft_prefix=data/tag.ft
minCount=3
minn=0
maxn=0
thread=4
dim=100
ws=5
epoch=5
utils/fasttext skipgram -input ${ft_in} -output ${ft_prefix} -lr 0.025 -dim ${dim} -ws ${ws} -epoch ${epoch} -minCount ${minCount} -neg 5 -loss ns -bucket 2000000 -minn ${minn} -maxn ${maxn} -thread ${thread} -t 1e-4 -lrUpdateRate 100
