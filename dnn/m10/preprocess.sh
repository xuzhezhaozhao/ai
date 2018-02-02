#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

ts=`date +%Y%m%d%H%M%S`

./preprocess_pctr.sh > log/pctr.log.${ts} 2>&1 &
./preprocess_xcbow.sh > log/xcbow.log.${ts} 2>&1 &

wait
wait
