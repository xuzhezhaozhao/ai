#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

ts=`date +%Y%m%d%H%M%S`
./fetchdata.sh > preprocess.log.${ts} 2>&1 && \
./preprocess_only_video.sh >> preprocess.log.${ts} 2>&1 && \
./sendupdate.sh >> preprocess.log.${ts} 2>&1
