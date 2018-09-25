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
train_ws=20
batch_size=32
eval_batch_size=8192
max_train_steps=-1
max_eval_steps=-1
max_eval_steps_on_train_dataset=40000
epoch=1
hidden_units=""
prefetch_size=20000
shuffle_batch=1
shuffle_size=50000
save_summary_steps=1000
save_checkpoints_secs=600
keep_checkpoint_max=3
log_step_count_steps=1000
log_step_count_secs=30
remove_model_dir=1
dropout=0.0
map_num_parallel_calls=1
inference_actions_len=100
inference_num_targets=200
# 'default', 'multi_thread'
train_parallel_mode='multi_thread'
train_num_parallel=4
# 'ada', 'sgd', 'adadelta', 'adam', 'rmsprop', 'momentum', 'ftrl'
optimizer_type='ftrl'
optimizer_epsilon=0.00001
optimizer_adadelta_rho=0.95
optimizer_adam_beta1=0.9
optimizer_adam_beta2=0.999
optimizer_rmsprop_decay=0.9
optimizer_rmsprop_momentum=0.0
optimizer_rmsprop_centered=0  # bool value
optimizer_momentum_momentum=0.99
optimizer_momentum_use_nesterov=0 # bool value
optimizer_ftrl_lr_power=-0.5
optimizer_ftrl_initial_accumulator_value=0.1
optimizer_ftrl_l1_regularization=0.0001
optimizer_ftrl_l2_regularization=0.0
optimizer_ftrl_l2_shrinkage_regularization=0.0
clip_gradients=0 # bool value
clip_gradients_norm=1.0
l2_regularizer=0.0
use_early_stopping=0
auc_num_thresholds=1000
optimizer_exponential_decay_steps=10000
optimizer_exponential_decay_rate=0.96
optimizer_exponential_decay_staircase=0  # bool value
evaluation_every_secs=10

min_count=10
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
    --max_eval_steps ${max_eval_steps} \
    --max_eval_steps_on_train_dataset ${max_eval_steps_on_train_dataset} \
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
    --log_step_count_secs ${log_step_count_secs} \
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
    --auc_num_thresholds ${auc_num_thresholds} \
    --optimizer_exponential_decay_steps ${optimizer_exponential_decay_steps} \
    --optimizer_exponential_decay_rate ${optimizer_exponential_decay_rate} \
    --optimizer_exponential_decay_staircase ${optimizer_exponential_decay_staircase} \
    --optimizer_ftrl_lr_power ${optimizer_ftrl_lr_power} \
    --optimizer_ftrl_initial_accumulator_value ${optimizer_ftrl_initial_accumulator_value} \
    --optimizer_ftrl_l1_regularization ${optimizer_ftrl_l1_regularization} \
    --optimizer_ftrl_l2_regularization ${optimizer_ftrl_l2_regularization} \
    --optimizer_ftrl_l2_shrinkage_regularization ${optimizer_ftrl_l2_shrinkage_regularization} \
    --evaluation_every_secs ${evaluation_every_secs}
