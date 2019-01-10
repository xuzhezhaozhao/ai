#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

clean_dir=$1
keep_max=3

echo "clean ${clean_dir} ..."
echo "keep_max = ${keep_max}"

keep=0
files=$(ls ${clean_dir} | sort -r)
for file in ${files}
do
    keep=$((keep+1))
    echo "check ${clean_dir}/${file} ..."
    if [[ ${keep} -gt ${keep_max} ]]; then
        echo "rm -rf ${clean_dir}/${file} ..."
        rm -rf ${clean_dir}/${file}
    fi
done

echo "clean ${clean_dir} OK"
