#! /usr/bin/env bash

set -e
keywords=(`cat $1`)

mkdir -p data
touch data/neg.txt
for key in ${keywords[@]}
do
    echo $key
    mkdir -p data/${key}
    grep -w ${key} ./kd_video_comments.csv > data/${key}/neg.txt

    cat data/${key}/neg.txt >> data/neg.txt
done
sed -i 's/^.*\t//g' data/neg.txt
cat data/neg.txt | sort | uniq > data/neg_uniq.txt
shuf data/neg_uniq.txt -o data/neg_uniq_shuf.txt
