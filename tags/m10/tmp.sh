#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}


rawdata_dir=raw_data
data_dir=data

input=${rawdata_dir}/records.csv

output=${data_dir}/`basename ${input}`

output_tags=${data_dir}/records_tags.in

echo 'shuf ...'


# fasttext
ft_in=${output_tags}.shuf
minCount=300
minn=0
maxn=0
thread=47
dim=100
ws=20
epoch=5
neg=5
lr=0.025

utils/fasttext skipgram \
    -input ${ft_in} \
    -output ${ft_in} \
    -lr ${lr} \
    -dim ${dim} \
    -ws ${ws} \
    -epoch ${epoch} \
    -minCount ${minCount} \
    -neg ${neg} \
    -loss ns \
    -bucket 2000000 \
    -minn ${minn} \
    -maxn ${maxn} \
    -thread ${thread} \
    -t 1e-4 \
    -lrUpdateRate 100  \
    && awk 'NR>2{print $1}' ${ft_in}.vec > ${ft_in}.dict
