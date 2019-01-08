#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

awk 'NR>1{print $1}' model/word2vec.vec > model/word2vec.dict
