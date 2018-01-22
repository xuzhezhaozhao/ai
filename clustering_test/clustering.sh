#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

mkdir -p cluster

parallel=47
ncluster=10
max_iter=300
tol=1e-4

preprocessed=data0/data.in.preprocessed.shuf
fast_model=${preprocessed}

python filter_vec.py \
       --input_fasttext_vec_file ${fast_model}.vec \
       --input_fasttext_subset_dict_file ${fast_model}.subset \
       --output_fasttext_subset_vec_file ${fast_model}.subset.vec

python clustering_article_and_videos.py \
    --input_video_vec_file ${preprocessed}.subset.vec \
    --input_all_vec_file ${preprocessed}.vec \
    --ncluster ${ncluster} \
    --output_cluster_file cluster/clusters.out \
    --njobs ${parallel} \
    --max_iter ${max_iter} \
    --tol ${tol}
