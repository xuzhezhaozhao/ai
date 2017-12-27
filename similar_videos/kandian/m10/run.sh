#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

ts=`date '+%s'`
./preprocess_only_video.sh > preprocess.log.${ts}
