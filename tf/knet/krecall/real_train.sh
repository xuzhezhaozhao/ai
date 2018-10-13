#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

# add Anaconda2
export ANACONDA2_ROOT=/usr/local/services/kd_anaconda2-1.0/lib/anaconda2
export PATH="${ANACONDA2_ROOT}/bin:$PATH"
export PYTHONPATH="${ANACONDA2_ROOT}/lib/python2.7/site-packages:$PYTHONPATH"

raw_data_dir=`pwd`/raw_data
model_dir=`pwd`/video_tab/model_dir
export_model_dir=`pwd`/video_tab/export_model_dir
dict_dir=`pwd`/video_tab/dict_dir
train_data_path=${raw_data_dir}/train_data.vt.in
eval_data_path=${raw_data_dir}/eval_data.vt.in
user_features_file=${raw_data_dir}/user_features.tsv

lr=0.5
embedding_dim=128
train_ws=20
train_lower_ws=1
min_count=100
t=1.0
batch_size=128
eval_batch_size=8192
num_sampled=10
epoch=1
hidden_units=""
prefetch_size=10000
shuffle_size=10000
max_train_steps=-1
save_summary_steps=100000000
save_checkpoints_secs=7200
log_step_count_steps=20000
recall_k=350
use_saved_dict=0
use_profile_hook=0
profile_steps=100000
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
predict_ws=50
sample_dropout=0.0
optimizer_type='ada'
tfrecord_file=${raw_data_dir}/train_data.vt.tfrecord
num_tfrecord_file=42
train_data_format='fasttext'  # 'tfrecord', 'fasttext'
map_num_parallel_calls=1
# 'default', 'train_op_parallel', 'multi_thread', 'multi_thread_v2'
train_parallel_mode='multi_thread'
train_num_parallel=16
use_batch_normalization=0
# 'exponential_decay', 'fasttext_decay', 'polynomial_decay', 'none'
sgd_lr_decay_type='fasttext_decay'
sgd_lr_decay_steps=100
sgd_lr_decay_rate=0.99
sgd_lr_decay_end_learning_rate=0.0001
sgd_lr_decay_power=1.0
use_clip_gradients=0
clip_norm=500.0
filter_with_rowkey_info=0
filter_with_rowkey_info_exposure_thr=10000
filter_with_rowkey_info_play=4000
filter_with_rowkey_info_e_play=2000
filter_with_rowkey_info_e_play_ratio_thr=0.3
rowkey_info_file=${raw_data_dir}/rowkey_info.json
normalize_nce_weights=0
normalize_embeddings=0
nce_loss_type='fasttext'  # 'word2vec', 'fasttext', 'default'
negative_sampler_type='fixed'  # fixed(better), log_uniform
use_user_features=0
use_age_feature=1
use_gender_feature=1
# 'indicator', 'numeric'
age_feature_type='indicator'
add_average_pooling=0
add_max_pooling=1
add_min_pooling=1
add_hierarchical_pooling=0
hierarchical_average_window=2
log_step_count_secs=300
evaluate_every_secs=100000000
max_eval_steps=-1
max_eval_steps_on_train_dataset=10000

if [[ ${train_data_format} == 'tfrecord' ]]; then
    dump_tfrecord_is_delete=1
    echo 'dump tfrecord ...'
    export LD_LIBRARY_PATH=./lib/:$LD_LIBRARY_PATH  # libtensorflow_framework.so

    ./utils/tfrecord_writer \
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
    --sgd_lr_decay_steps ${sgd_lr_decay_steps} \
    --sgd_lr_decay_rate ${sgd_lr_decay_rate} \
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
    --sgd_lr_decay_end_learning_rate ${sgd_lr_decay_end_learning_rate} \
    --sgd_lr_decay_power ${sgd_lr_decay_power} \
    --use_user_features ${use_user_features} \
    --user_features_file ${user_features_file} \
    --use_age_feature ${use_age_feature} \
    --use_gender_feature ${use_gender_feature} \
    --age_feature_type ${age_feature_type} \
    --add_average_pooling ${add_average_pooling} \
    --add_max_pooling ${add_max_pooling} \
    --add_min_pooling ${add_min_pooling} \
    --add_hierarchical_pooling ${add_hierarchical_pooling} \
    --hierarchical_average_window ${hierarchical_average_window} \
    --log_step_count_secs ${log_step_count_secs} \
    --evaluate_every_secs ${evaluate_every_secs} \
    --max_eval_steps ${max_eval_steps} \
    --max_eval_steps_on_train_dataset ${max_eval_steps_on_train_dataset}
