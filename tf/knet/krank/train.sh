#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir
fe_dir=`pwd`/fe_dir

rowkey_count_path=../../../data/krank_rowkey_count.csv
train_data_path=../../../data/krank_train_data.in
eval_data_path=../../../data/krank_eval_data.in
feature_manager_path=${fe_dir}/feature_manager.bin
lr=0.01
rowkey_embedding_dim=64
train_ws=50
batch_size=128
eval_batch_size=1024
max_train_steps=10000
epoch=5
hidden_units="256,256"
prefetch_size=20000
shuffle_batch=1
shuffle_size=20000
save_summary_steps=1000
save_checkpoints_secs=600
keep_checkpoint_max=3
log_step_count_steps=1000
use_profile_hook=0
profile_steps=500
remove_model_dir=0
dropout=0.0
map_num_parallel_calls=1
inference_actions_len=100
inference_num_targets=200
# 'default', 'train_op_parallel', 'multi_thread', 'multi_thread_v2'
train_parallel_mode='default'
train_num_parallel=8
optimizer_type='ada'  # 'ada', 'sgd', 'adadelta', 'adam', 'rmsprop', 'momentum'
optimizer_epsilon=0.00001
optimizer_adadelta_rho=0.95
optimizer_adam_beta1=0.9
optimizer_adam_beta2=0.999
optimizer_rmsprop_decay=0.9
optimizer_rmsprop_momentum=0.0
optimizer_rmsprop_centered=0  # bool value
optimizer_momentum_momentum=0.6
optimizer_momentum_use_nesterov=0 # bool value
clip_gradients=1 # bool value
clip_gradients_norm=5.0
l2_regularizer=0.0
use_early_stopping=0
early_stopping_start_delay_secs=120
early_stopping_throttle_secs=600

min_count=50
rowkey_dict_path=${fe_dir}/rowkey_dict.txt

if [[ ${remove_model_dir} == '1' ]]; then
    rm -rf ${model_dir}
fi

mkdir -p ${fe_dir}
echo "Preprocess features ..."
./fe/build/preprocess \
    ${rowkey_count_path} \
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
    --eval_batch_size ${eval_batch_size} \
    --max_train_steps ${max_train_steps} \
    --epoch ${epoch} \
    --hidden_units "${hidden_units}" \
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
    --rowkey_dict_path ${rowkey_dict_path} \
    --inference_actions_len ${inference_actions_len} \
    --inference_num_targets ${inference_num_targets} \
    --train_parallel_mode ${train_parallel_mode} \
    --train_num_parallel ${train_num_parallel} \
    --optimizer_type ${optimizer_type} \
    --optimizer_epsilon ${optimizer_epsilon} \
    --optimizer_adadelta_rho ${optimizer_adadelta_rho} \
    --optimizer_adam_beta1 ${optimizer_adam_beta1} \
    --optimizer_adam_beta2 ${optimizer_adam_beta2} \
    --optimizer_rmsprop_decay ${optimizer_rmsprop_decay} \
    --optimizer_rmsprop_momentum ${optimizer_rmsprop_momentum} \
    --optimizer_rmsprop_centered ${optimizer_rmsprop_centered} \
    --optimizer_momentum_momentum ${optimizer_momentum_momentum} \
    --optimizer_momentum_use_nesterov ${optimizer_momentum_use_nesterov} \
    --clip_gradients ${clip_gradients} \
    --clip_gradients_norm ${clip_gradients_norm} \
    --l2_regularizer ${l2_regularizer} \
    --use_early_stopping ${use_early_stopping} \
    --early_stopping_start_delay_secs ${early_stopping_start_delay_secs} \
    --early_stopping_throttle_secs ${early_stopping_throttle_secs}
