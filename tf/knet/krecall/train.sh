#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir

if [[ $# != 0 ]] ; then
    if [[ $1 == "user" ]]; then
        echo "Use new train data ..."
        train_data_path=../../../data/train_data.in.new
        eval_data_path=../../../data/eval_data.in.new
    else
        echo "Usage: [user]"
        exit -1
    fi
else
    train_data_path=../../../data/train_data.in
    eval_data_path=../../../data/eval_data.in
fi

lr=1.0
embedding_dim=100
train_ws=20
train_lower_ws=1
min_count=30
t=0.025
batch_size=64
eval_batch_size=1024
num_sampled=5
epoch=5
hidden_units=""
prefetch_size=10000
shuffle_size=10000
max_train_steps=-1
max_eval_steps=-1
max_eval_steps_in_train=50
save_summary_steps=100
save_checkpoints_secs=600
log_step_count_steps=1000
recall_k=10
dict_dir=`pwd`/dict_dir
use_saved_dict=0
use_profile_hook=0
profile_steps=1000
root_ops_path=lib/
remove_model_dir=1
optimize_level=1
receive_ws=100
use_subset=1
dropout=0.0
ntargets=1
chief_lock=${model_dir}/chief.lock
max_distribute_train_steps=-1
train_nce_biases=0
shuffle_batch=1
predict_ws=20
sample_dropout=0.0
# 'adagrad', 'sgd', 'adadelta', 'adam', 'rmsprop', 'momentum', 'ftrl'
optimizer_type='ftrl'
tfrecord_file='../../../data/train_data.tfrecord'
num_tfrecord_file=2
train_data_format='fasttext'  # 'tfrecord', 'fasttext'
map_num_parallel_calls=1
# 'default', 'multi_thread'
train_parallel_mode='multi_thread'
train_num_parallel=4
use_batch_normalization=1
# 'exponential_decay', 'fasttext_decay', 'polynomial_decay', 'none'
sgd_lr_decay_type='fasttext_decay'
use_clip_gradients=0
clip_norm=500.0
filter_with_rowkey_info=0
filter_with_rowkey_info_exposure_thr=10000
filter_with_rowkey_info_play=100
filter_with_rowkey_info_e_play=100
filter_with_rowkey_info_e_play_ratio_thr=0.3
rowkey_info_file=""
normalize_nce_weights=0
normalize_embeddings=0
nce_loss_type='fasttext'  # 'word2vec', 'fasttext', 'default'
negative_sampler_type='fixed'  # fixed(better), log_uniform
use_user_features=0
user_features_file="../../../data/user_features.tsv"
use_age_feature=1
use_gender_feature=1
# 'indicator', 'numeric'
age_feature_type='numeric'
add_average_pooling=0
add_max_pooling=1
add_min_pooling=0
add_hierarchical_pooling=0
add_attention_layer=0
hierarchical_average_window=5
attention_size=500
log_step_count_secs=10
evaluate_every_secs=500000
max_eval_steps_on_train_dataset=1000
optimizer_epsilon=0.00001
optimizer_adadelta_rho=0.95
optimizer_adam_beta1=0.9
optimizer_adam_beta2=0.999
optimizer_rmsprop_decay=0.9
optimizer_rmsprop_momentum=0.01
optimizer_rmsprop_centered=0  # bool value
optimizer_momentum_momentum=0.9
optimizer_momentum_use_nesterov=0 # bool value
optimizer_ftrl_lr_power=-6.5
optimizer_ftrl_initial_accumulator_value=1.0
optimizer_ftrl_l1_regularization=0.000
optimizer_ftrl_l2_regularization=0.00001
optimizer_ftrl_l2_shrinkage_regularization=0.0
optimizer_exponential_decay_steps=1000
optimizer_exponential_decay_rate=0.5
optimizer_exponential_decay_staircase=0  # bool value
log_per_lines=100000
cpp_log_level=1  # 0: all
tf_log_level='INFO'  # DEBUG, INFO, ERROR, FATAL

if [[ ${train_data_format} == 'tfrecord' ]]; then
    dump_tfrecord_is_delete=1
    echo 'dump tfrecord ...'
    export LD_LIBRARY_PATH=./lib/:$LD_LIBRARY_PATH  # libtensorflow_framework.so

    ./ops/fasttext/tfrecord_writer \
        --tfrecord_file ${tfrecord_file} \
        --ws ${train_ws} \
        --lower_ws ${train_lower_ws} \
        --min_count ${min_count} \
        -t ${t} \
        --ntargets ${ntargets} \
        --sample_dropout ${sample_dropout} \
        --train_data_path ${train_data_path} \
        --dict_dir ${dict_dir} \
        --threads ${num_tfrecord_file} \
        --is_delete ${dump_tfrecord_is_delete} \
        --use_saved_dict ${use_saved_dict}
    echo 'dump tfrecord OK'
fi


python main.py \
    --train_data_path ${train_data_path} \
    --eval_data_path ${eval_data_path} \
    --lr ${lr} \
    --embedding_dim ${embedding_dim} \
    --train_ws ${train_ws} \
    --train_lower_ws ${train_lower_ws} \
    --min_count ${min_count} \
    --t ${t} \
    --verbose 2 \
    --min_count_label 50 \
    --label "__label__" \
    --batch_size ${batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --num_sampled ${num_sampled} \
    --epoch ${epoch} \
    --hidden_units "${hidden_units}" \
    --model_dir ${model_dir} \
    --export_model_dir ${export_model_dir} \
    --prefetch_size ${prefetch_size} \
    --shuffle_size ${shuffle_size} \
    --max_train_steps ${max_train_steps} \
    --max_eval_steps ${max_eval_steps} \
    --max_eval_steps_in_train ${max_eval_steps_in_train} \
    --save_summary_steps ${save_summary_steps} \
    --save_checkpoints_secs ${save_checkpoints_secs} \
    --keep_checkpoint_max 2 \
    --log_step_count_steps ${log_step_count_steps} \
    --recall_k ${recall_k} \
    --dict_dir ${dict_dir} \
    --use_saved_dict ${use_saved_dict} \
    --use_profile_hook ${use_profile_hook} \
    --profile_steps ${profile_steps} \
    --root_ops_path ${root_ops_path} \
    --remove_model_dir ${remove_model_dir} \
    --optimize_level ${optimize_level} \
    --receive_ws ${receive_ws} \
    --use_subset ${use_subset} \
    --dropout ${dropout} \
    --ntargets ${ntargets} \
    --chief_lock ${chief_lock} \
    --max_distribute_train_steps ${max_distribute_train_steps} \
    --train_nce_biases ${train_nce_biases} \
    --shuffle_batch ${shuffle_batch} \
    --predict_ws ${predict_ws} \
    --sample_dropout ${sample_dropout} \
    --optimizer_type ${optimizer_type} \
    --tfrecord_file ${tfrecord_file} \
    --num_tfrecord_file ${num_tfrecord_file} \
    --train_data_format ${train_data_format} \
    --map_num_parallel_calls ${map_num_parallel_calls} \
    --train_parallel_mode ${train_parallel_mode} \
    --train_num_parallel ${train_num_parallel} \
    --use_batch_normalization ${use_batch_normalization} \
    --sgd_lr_decay_type ${sgd_lr_decay_type} \
    --use_clip_gradients ${use_clip_gradients} \
    --clip_norm ${clip_norm} \
    --filter_with_rowkey_info ${filter_with_rowkey_info} \
    --filter_with_rowkey_info_exposure_thr ${filter_with_rowkey_info_exposure_thr} \
    --filter_with_rowkey_info_play ${filter_with_rowkey_info_play} \
    --filter_with_rowkey_info_e_play ${filter_with_rowkey_info_e_play} \
    --filter_with_rowkey_info_e_play_ratio_thr ${filter_with_rowkey_info_e_play_ratio_thr} \
    --rowkey_info_file "${rowkey_info_file}" \
    --normalize_nce_weights ${normalize_nce_weights} \
    --normalize_embeddings ${normalize_embeddings} \
    --nce_loss_type ${nce_loss_type} \
    --negative_sampler_type ${negative_sampler_type} \
    --use_user_features ${use_user_features} \
    --user_features_file ${user_features_file} \
    --use_age_feature ${use_age_feature} \
    --use_gender_feature ${use_gender_feature} \
    --age_feature_type ${age_feature_type} \
    --add_average_pooling ${add_average_pooling} \
    --add_max_pooling ${add_max_pooling} \
    --add_min_pooling ${add_min_pooling} \
    --add_hierarchical_pooling ${add_hierarchical_pooling} \
    --add_attention_layer ${add_attention_layer} \
    --hierarchical_average_window ${hierarchical_average_window} \
    --attention_size ${attention_size} \
    --log_step_count_secs ${log_step_count_secs} \
    --evaluate_every_secs ${evaluate_every_secs} \
    --max_eval_steps_on_train_dataset ${max_eval_steps_on_train_dataset} \
    --optimizer_epsilon ${optimizer_epsilon} \
    --optimizer_adadelta_rho ${optimizer_adadelta_rho} \
    --optimizer_adam_beta1 ${optimizer_adam_beta1} \
    --optimizer_adam_beta2 ${optimizer_adam_beta2} \
    --optimizer_rmsprop_decay ${optimizer_rmsprop_decay} \
    --optimizer_rmsprop_momentum ${optimizer_rmsprop_momentum} \
    --optimizer_rmsprop_centered ${optimizer_rmsprop_centered} \
    --optimizer_momentum_momentum ${optimizer_momentum_momentum} \
    --optimizer_momentum_use_nesterov ${optimizer_momentum_use_nesterov} \
    --optimizer_ftrl_lr_power ${optimizer_ftrl_lr_power} \
    --optimizer_ftrl_initial_accumulator_value ${optimizer_ftrl_initial_accumulator_value} \
    --optimizer_ftrl_l1_regularization ${optimizer_ftrl_l1_regularization} \
    --optimizer_ftrl_l2_regularization ${optimizer_ftrl_l2_regularization} \
    --optimizer_ftrl_l2_shrinkage_regularization ${optimizer_ftrl_l2_shrinkage_regularization} \
    --optimizer_exponential_decay_steps ${optimizer_exponential_decay_steps} \
    --optimizer_exponential_decay_rate ${optimizer_exponential_decay_rate} \
    --optimizer_exponential_decay_staircase ${optimizer_exponential_decay_staircase} \
    --log_per_lines ${log_per_lines} \
    --cpp_log_level ${cpp_log_level} \
    --tf_log_level ${tf_log_level} \
    --attention_size ${attention_size}
