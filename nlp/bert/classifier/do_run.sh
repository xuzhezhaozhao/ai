#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source /usr/local/services/kd_anaconda2-1.0/anaconda2_profile

python ${MYDIR}/run_classifier.py $@
