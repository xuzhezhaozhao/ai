#! /usr/bin/env bash

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

chmod 600 rsync.passwd

fetch_files=( \
    "data.in.tf.dict" \
    "data.in.tf.predicts" \
    "data.in.tf.predicts.pctr" \
    "data.in.tf.watched" \
    "data.in.tf.watched.pctr" \
    "data.in.tf.vec" \
    "data.in.preprocessed.watched_ratio"
)

echo "fetch file list: ${fetch_files[@]}"
for filename in ${fetch_files[@]}
do
    if [ -f "$filename" ]; then
      echo "backup ${filename} ..."
      mkdir -p bak
      cp ${filename} bak/${filename}.bak
    fi
    rsync -azvcP --password-file=rsync.passwd rsync://user_00@100.115.1.220:49000/service/dnn/ynet/data/${filename} ./
done
