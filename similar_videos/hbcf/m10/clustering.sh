#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

final_data_dir=data1
input=${final_data_dir}/data.in
preprocessed=${input}.preprocessed

mkdir -p cluster

parallel=47
ncluster=2000
max_iter=300
tol=1e-4
python clustering_videos.py \
    --input_vec_file ${preprocessed}.shuf.vec \
    --ncluster ${ncluster} \
    --output_cluster_file cluster/clusters.out \
    --njobs ${parallel} \
    --max_iter ${max_iter} \
    --tol ${tol}
