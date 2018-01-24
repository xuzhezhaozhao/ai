#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

mkdir -p cluster

parallel=47
ncluster=1000
max_iter=300
tol=1e-4

preprocessed=data0/data.in.preprocessed.shuf
fast_model=${preprocessed}

cp -p ${fast_model}.subset .
cp -p ${fast_model}.vec .

prefix=data.in.preprocessed.shuf
python filter_vec.py \
       --input_fasttext_vec_file ${prefix}.vec \
       --input_fasttext_subset_dict_file ${prefix}.subset \
       --output_fasttext_subset_vec_file ${prefix}.subset.vec

python clustering_article_and_videos.py \
    --input_video_vec_file ${prefix}.subset.vec \
    --input_all_vec_file ${prefix}.vec \
    --ncluster ${ncluster} \
    --output_cluster_file cluster/clusters.out \
    --njobs ${parallel} \
    --max_iter ${max_iter} \
    --tol ${tol}

/data/utils/sendupdate -modid=740097 -cmdid=3932160
