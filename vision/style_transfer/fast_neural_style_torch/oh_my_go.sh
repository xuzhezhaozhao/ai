#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir
