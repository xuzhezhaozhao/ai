#! /usr/bin/env bash
model=$1
k=$2
input=$3
output=$4
~/fastText/fasttext nn ${model} ${k} < ${input} > ${output}
