#! /usr/bin/env bash

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

find log/* -mtime +2 -exec rm -rf {} \;
ts=`date +%Y%m%d%H%M%S`

vt_log=log/vt.log.${ts}

(./fetchdata_vt.sh && ./preprocess_vt.sh) > ${vt_log} 2>&1
