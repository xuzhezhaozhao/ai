#! /usr/bin/env bash
input=$1
output=$2
num=$3
awk '{print $1}' ${input} | shuf -n ${num} > ${output}
