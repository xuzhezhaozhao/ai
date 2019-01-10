#! /usr/bin/env bash

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

find log/* -mtime +60 -exec rm -rf {} \;
ts=`date +%Y%m%d%H%M%S`

log_file=log/log.${ts}

( \
./fetchdata.sh \
&& ./real_train.sh \
) > ${log_file} 2>&1
