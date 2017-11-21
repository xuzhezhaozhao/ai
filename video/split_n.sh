#! /usr/env bash
input=$1
nsplits=$2
prefix=${input}.

split -d -n l/${nsplits} ${input} ${prefix}
