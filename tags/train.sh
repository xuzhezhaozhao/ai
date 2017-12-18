#! /usr/bin/env bash

ft_in=data/tags.in.shuf
ft_in_raw=data/tags.raw.in.shuf

minCount=10
minn=0
maxn=0
thread=4
dim=100
ws=5
epoch=5
neg=5
lr=0.025

for input in ${ft_in} ${ft_in_raw}; do

utils/fasttext skipgram -input ${input} -output ${input} -lr ${lr} -dim ${dim} -ws ${ws} -epoch ${epoch} -minCount ${minCount} -neg ${neg} -loss ns -bucket 2000000 -minn ${minn} -maxn ${maxn} -thread ${thread} -t 1e-4 -lrUpdateRate 100  && awk 'NR>2{print $1}' ${input}.vec > ${input}.dict

done
