#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

data_dir=hbcf/data1
input=${data_dir}/data.in
preprocessed=${input}.preprocessed
fast_model=${preprocessed}.shuf
nn_k=100
click_bias=100

python add_play_ratio.py --input_play_ratio_file ${preprocessed}.play_raito --input_nn_result_file ${fast_model}.result.raw --output_result_file ${fast_model}.result --nn_k ${nn_k} --nn_score_weight 0.3
# L5
./utils/sendupdate -modid=907457 -cmdid=65536 -data=0

sleep 600

python add_valid_click.py --input_click_file ${preprocessed}.video_click --input_nn_result_file ${fast_model}.result.raw --output_result_file ${fast_model}.result --nn_k ${nn_k} --nn_score_weight 0.0 --click_bias ${click_bias}
# experiment L5
./utils/sendupdate -modid=907457 -cmdid=65537 -data=2
