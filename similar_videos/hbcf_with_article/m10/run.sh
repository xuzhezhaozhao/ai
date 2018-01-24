#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

mkdir -p log
find log -mtime +2 -exec rm -rf {} \;

ts=`date +%Y%m%d%H%M%S`
logfile=log/preprocess.log.${ts}

./fetchdata.sh > ${logfile} 2>&1 && \
./preprocess.sh >> ${logfile} 2>&1 && \
./multi_version.sh >> ${logfile} 2>&1
