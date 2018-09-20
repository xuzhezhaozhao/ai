#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir

train_data_path=../../../data/krank_train_data.in
eval_data_path=../../../data/krank_eval_data.in
feature_manager_path=./feature_manager.bin
lr=0.01
rowkey_embedding_dim=128
train_ws=50
batch_size=1024
max_train_steps=-1
epoch=5
hidden_units="200,200,200"
prefetch_size=100000
shuffle_batch=1
shuffle_size=100000
save_summary_steps=100
save_checkpoints_secs=600
keep_checkpoint_max=3
log_step_count_steps=100
use_profile_hook=0
profile_steps=500
remove_model_dir=0
dropout=0.5
map_num_parallel_calls=1

min_count=50
rowkey_dict_path=./rowkey_dict.txt

echo "Preprocess features ..."
./fe/build/preprocess \
    ${train_data_path} \
    ${min_count} \
    ${feature_manager_path} \
    ${rowkey_dict_path}

python main.py \
    --train_data_path ${train_data_path} \
    --eval_data_path ${eval_data_path} \
    --feature_manager_path ${feature_manager_path} \
    --lr ${lr} \
    --rowkey_embedding_dim ${rowkey_embedding_dim} \
    --train_ws ${train_ws} \
    --batch_size ${batch_size} \
    --max_train_steps ${max_train_steps} \
    --epoch ${epoch} \
    --hidden_units ${hidden_units} \
    --model_dir ${model_dir} \
    --export_model_dir ${export_model_dir} \
    --prefetch_size ${prefetch_size} \
    --shuffle_batch ${shuffle_batch} \
    --shuffle_size ${shuffle_size} \
    --save_summary_steps ${save_summary_steps} \
    --save_checkpoints_secs ${save_checkpoints_secs} \
    --keep_checkpoint_max ${keep_checkpoint_max} \
    --log_step_count_steps ${log_step_count_steps} \
    --use_profile_hook ${use_profile_hook} \
    --profile_steps ${profile_steps} \
    --remove_model_dir ${remove_model_dir} \
    --dropout ${dropout} \
    --map_num_parallel_calls ${map_num_parallel_calls} \
    --rowkey_dict_path ${rowkey_dict_path}
