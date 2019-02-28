#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

# speedup 2~5x
# https://github.com/facebookresearch/faiss/wiki/Troubleshooting
# https://github.com/facebookresearch/faiss/issues/53
export OMP_WAIT_POLICY=PASSIVE

./do_kmeans $@
