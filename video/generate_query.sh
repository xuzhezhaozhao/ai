#! /usr/bin/env bash
input=$1
output=$2
awk '{print $1}' ${input} > ${output}
