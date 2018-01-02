#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

mkdir -p log

ts=`date +%Y%m%d%H%M%S`
logfile=log/preprocess.log.${ts}
./fetchdata.sh > ${logfile} 2>&1 && \
./preprocess_only_video.sh >> ${logfile} 2>&1 && \
./sendupdate.sh >> ${logfile} 2>&1
