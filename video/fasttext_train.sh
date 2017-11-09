#! /usr/bin/env bash
input=$1
output=$2
~/fastText/fasttext skipgram -input ${input} -output ${output} -lr 0.025 -dim 100 -ws 5 -epoch 1 -minCount 1 -neg 5 -loss ns -bucket 2000000 -minn 0 -maxn 0 -thread 4 -t 1e-4 -lrUpdateRate 100
