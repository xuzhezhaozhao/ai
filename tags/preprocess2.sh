#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

input=$1

echo 'delete csv file header ...'
#sed -n '1p' ${input} > ${input}.header
#sed -i "1d" ${input}

echo "sort csv file with 1st field ..."
sorted_file=${input}.sorted
mkdir -p tmp_sort/
sort -T tmp_sort/ -t ',' -k 1 --parallel=4 ${input} -o ${sorted_file}
rm -rf tmp_sort/
