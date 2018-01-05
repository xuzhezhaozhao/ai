#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

input=data.in
output=${input}.fasttext

cat data/* > ${input}


# max items per user
klimit=5000
# fasttext min count
minCount=100

sorted_file=${input}.sorted

echo "sort input file ..."

echo "transform sorted file ..."
./transform.py ${sorted_file} ${output} ${klimit}

echo "fastText train ..."
fast_model=${output}.model
./fasttext skipgram -input ${output} -output ${fast_model} -lr 0.025\
  -dim 100 -ws 20 -epoch 5 -minCount ${minCount} -neg 5 -loss ns -bucket 2000000\
  -minn 0 -maxn 0 -thread 47 -t 1e-4 -lrUpdateRate 100

echo "generate query list ..."
awk '{print $1}' ${fast_model}.vec > ${fast_model}.query
echo "remove first two lines of query list ..."
sed -i "1d" ${fast_model}.query
sed -i "1d" ${fast_model}.query

echo "split query list ..."
split -d -n l/36 ${fast_model}.query ${fast_model}.query.

echo "fastText nn ..."
FASTTEST=./fasttext
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.00 > ${fast_model}.result.00 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.01 > ${fast_model}.result.01 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.02 > ${fast_model}.result.02 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.03 > ${fast_model}.result.03 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.04 > ${fast_model}.result.04 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.05 > ${fast_model}.result.05 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.06 > ${fast_model}.result.06 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.07 > ${fast_model}.result.07 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.08 > ${fast_model}.result.08 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.09 > ${fast_model}.result.09 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.10 > ${fast_model}.result.10 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.11 > ${fast_model}.result.11 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.12 > ${fast_model}.result.12 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.13 > ${fast_model}.result.13 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.14 > ${fast_model}.result.14 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.15 > ${fast_model}.result.15 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.16 > ${fast_model}.result.16 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.17 > ${fast_model}.result.17 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.18 > ${fast_model}.result.18 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.19 > ${fast_model}.result.19 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.20 > ${fast_model}.result.20 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.21 > ${fast_model}.result.21 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.22 > ${fast_model}.result.22 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.23 > ${fast_model}.result.23 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.24 > ${fast_model}.result.24 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.25 > ${fast_model}.result.25 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.26 > ${fast_model}.result.26 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.27 > ${fast_model}.result.27 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.28 > ${fast_model}.result.28 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.29 > ${fast_model}.result.29 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.30 > ${fast_model}.result.30 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.31 > ${fast_model}.result.31 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.32 > ${fast_model}.result.32 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.33 > ${fast_model}.result.33 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.34 > ${fast_model}.result.34 &
${FASTTEST} nn ${fast_model}.bin 50 100 < ${fast_model}.query.35 > ${fast_model}.result.35 &
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
wait
cat ${fast_model}.result.* > ${fast_model}.result

rm -rf ${sorted_file}
rm -rf ${fast_model}.query.*
rm -rf ${fast_model}.result.*

./sendupdate -modid=869569 -cmdid=65536
