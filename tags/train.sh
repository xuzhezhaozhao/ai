#! /usr/bin/env bash

ft_in=data/tags.ft.in
ft_prefix=data/tags.ft

minCount=3
minn=0
maxn=0
thread=2
dim=100
ws=5
epoch=5
utils/fasttext skipgram -input ${ft_in} -output ${ft_prefix} -lr 0.025 -dim ${dim} -ws ${ws} -epoch ${epoch} -minCount ${minCount} -neg 5 -loss ns -bucket 2000000 -minn ${minn} -maxn ${maxn} -thread ${thread} -t 1e-4 -lrUpdateRate 100

awk 'NR>2{print $1}' tags.ft.vec > ${ft_prefix}.dict
