#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

/data/utils/sendupdate -modid=900545 -cmdid=65536 > log/sendupdate_dnn.log 2>&1 || /data/utils/sendupdate -modid=900545 -cmdid=65536 > log/sendupdate_dnn.log 2>&1
