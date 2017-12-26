#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

input=data/data.in

echo "sort csv file with 1st field ..."
sorted_file=${input}.sorted
mkdir -p tmp_sort/
sort -T tmp_sort/ -t ',' -k 1 --parallel=4 ${input} -o ${sorted_file}
rm -rf tmp_sort/

preprocessed=${input}.preprocessed
echo "transform sorted file to fastText format ..."
./preprocess/build/src/preprocess \
    -raw_input=${sorted_file} \
    -with_header=false \
    -only_video=true \
    -interval=1000000 \
    -output_user_watched_file=${preprocessed} \
    -user_min_watched=10 \
    -user_max_watched=1024 \
    -user_abnormal_watched_thr=2048 \
    -supress_hot_arg1=8 \
    -supress_hot_arg2=3 \
    -user_effective_watched_time_thr=20 \
    -user_effective_watched_ratio_thr=0.3


echo "fastText train ..."
# fasttext args
minCount=50
dim=100
watched_size=20
max_per_user=100
ws=10
epoch=5
neg=5
thread=4
fast_output=${input}
utils/fasttext skipgram -input ${preprocessed} -output ${fast_output} -lr 0.025\
  -dim ${dim} -ws ${ws} -epoch ${epoch} -minCount ${minCount} -neg ${neg} -loss ns -bucket 2000000\
  -minn 0 -maxn 0 -thread ${thread} -t 1e-4 -lrUpdateRate 100

tf_input=${input}.tf
python utils/vec2binary.py --input ${fast_output}.vec --output ${tf_input}.vec --output_dict_file ${tf_input}.dict
python utils/filter_transform.py --input ${preprocessed} --output ${tf_input}.records --dictfile ${tf_input}.dict
python utils/records2binary.py --input_records ${tf_input}.records --output_watched ${tf_input}.watched --output_predicts ${tf_input}.predicts --input_dict_file ${tf_input}.dict --watched_size ${watched_size} --max_per_user ${max_per_user}
