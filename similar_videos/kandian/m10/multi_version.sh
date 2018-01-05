#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

data_dir=data1
input=${data_dir}/data.in
preprocessed=${input}.preprocessed
fast_model=${preprocessed}.shuf
nn_k=100

echo "algo 1 ..."
python add_play_ratio.py --input_play_ratio_file ${preprocessed}.play_raito --input_nn_result_file ${fast_model}.result.raw --output_result_file ${fast_model}.result --nn_k ${nn_k} --nn_score_weight 0.5
/data/utils/sendupdate -test=true -ip=10.56.90.138 -port=52003 -data=1

sleep 300
echo "algo 2 ..."
python add_play_ratio.py --input_play_ratio_file ${preprocessed}.play_raito --input_nn_result_file ${fast_model}.result.raw --output_result_file ${fast_model}.result --nn_k ${nn_k} --nn_score_weight 0.3
/data/utils/sendupdate -test=true -ip=10.56.60.28 -port=52003 -data=2

sleep 300
echo "algo 3 ..."
python add_play_ratio.py --input_play_ratio_file ${preprocessed}.play_raito --input_nn_result_file ${fast_model}.result.raw --output_result_file ${fast_model}.result --nn_k ${nn_k} --nn_score_weight 0
/data/utils/sendupdate -test=true -ip=100.108.26.169 -port=52003 -data=3
