#! /usr/bin/env bash

# input tdw table
input=$1
output=$1.fasttext
# max items per user
klimit=$2
# fasttext min count
minCount=$3

tmp_file=${input}.tmp
sorted_file=${input}.sorted

sed "1d" ${input} > ${tmp_file}
sort -t ',' -k 4 ${tmp_file} > ${sorted_file}

./transform.py ${sorted_file} ${output} ${klimit}

fast_model=${output}.model
~/fastText/fasttext skipgram -input ${output} -output ${fast_model} -lr 0.025 -dim 100 -ws 5 -epoch 2 -minCount ${minCount} -neg 5 -loss ns -bucket 2000000 -minn 0 -maxn 0 -thread 6 -t 1e-4 -lrUpdateRate 100

awk '{print $1}' ${fast_model}.vec > ${fast_model}.query
sed -i "1d" ${fast_model}.query
sed -i "2d" ${fast_model}.query



~/fastText/fasttext nn ${fast_model}.bin 50 30 < ${fast_model}.query > ${fast_model}.result
