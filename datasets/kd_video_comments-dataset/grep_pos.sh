#! /usr/bin/env bash

set -e
keywords=(`cat $1`)
N=6000

cmd="cat ./kd_video_comments.csv"
for key in ${keywords[@]}
do
    cmd+="|grep"" ""-v"" "${key}
done
echo 'cmd: ' ${cmd}
eval ${cmd} > data/pos_all.txt
shuf data/pos_all.txt | head -n ${N} > data/pos.txt
sed -i 's/^.*\t//g' data/pos.txt
cat data/pos.txt | sort | uniq > data/pos_uniq.txt
shuf data/pos_uniq.txt -o data/pos_uniq_shuf.txt
