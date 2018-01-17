#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

/data/utils/sendupdate -modid=918081 -cmdid=65536 > log/sendupdate_pctr.log 2>&1 
/data/utils/sendupdate -modid=918081 -cmdid=65536 >> log/sendupdate_pctr.log 2>&1
