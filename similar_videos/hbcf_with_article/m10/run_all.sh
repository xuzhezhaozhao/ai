#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

rm -rf log
mkdir -p log

ts=`date +%Y%m%d%H%M%S`
logfile=log/preprocess_all.log.${ts}

./fetchdata_all.sh > ${logfile} 2>&1 && \
./preprocess_all.sh >> ${logfile} 2>&1 && \
./sendupdate_all.sh >> ${logfile} 2>&1
