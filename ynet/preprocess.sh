#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

input=$1
output=${input}.fasttext

# min items per user
kmin=50
# fasttext min count
minCount=1

echo 'delete csv file header ...'
sed -n '1p' ${input} > ${input}.header
sed -i "1d" ${input}

echo "sort csv file with 1st field ..."
sorted_file=${input}.sorted
mkdir -p tmp_sort/
sort -T tmp_sort/ -t ',' -k 1 --parallel=4 ${input} -o ${sorted_file}
rm -rf tmp_sort/

echo "transform sorted file to fastText format ..."
./transform.py ${sorted_file} ${output} ${kmin}

echo "fastText train ..."
fast_model=${output}.model
./fasttext skipgram -input ${output} -output ${fast_model} -lr 0.025\
  -dim 256 -ws 5 -epoch 1 -minCount ${minCount} -neg 5 -loss ns -bucket 2000000\
  -minn 0 -maxn 0 -thread 4 -t 1e-4 -lrUpdateRate 100
