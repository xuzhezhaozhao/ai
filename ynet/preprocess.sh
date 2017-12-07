#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

input=$1

# min items per user
kmin=21

# fasttext args
minCount=1
dim=256

watched_size=20
max_per_user=1
ws=5
epoch=1
neg=5
thread=4


#echo 'delete csv file header ...'
#sed -n '1p' ${input} > ${input}.header
#sed -i "1d" ${input}

echo "sort csv file with 1st field ..."
sorted_file=${input}.sorted
mkdir -p tmp_sort/
sort -T tmp_sort/ -t ',' -k 1 --parallel=4 ${input} -o ${sorted_file}
rm -rf tmp_sort/

preprocessed=${input}.preprocessed
echo "transform sorted file to fastText format ..."
python utils/transform.py ${sorted_file} ${preprocessed} ${kmin}

echo "fastText train ..."
fast_output=${input}
utils/fasttext skipgram -input ${preprocessed} -output ${fast_output} -lr 0.025\
  -dim ${dim} -ws ${ws} -epoch ${epoch} -minCount ${minCount} -neg ${neg} -loss ns -bucket 2000000\
  -minn 0 -maxn 0 -thread ${thread} -t 1e-4 -lrUpdateRate 100

tf_input=${input}.tf
python utils/vec2binary.py --input ${fast_output}.vec --output ${tf_input}.vec --output_dict_file ${tf_input}.dict
python utils/records2binary.py --input_records ${preprocessed} --output_watched ${tf_input}.watched --output_predicts ${tf_input}.predicts --input_dict_file ${tf_input}.dict --watched_size ${watched_size} --max_per_user ${max_per_user}
