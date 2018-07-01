#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

echo 'TF_CONFIG = ' ${TF_CONFIG}

MODEL_DIR=`pwd`/model_dir
EXPORT_MODEL_DIR=`pwd`/export_model_dir

if [ $# != 0 ] ; then
    train_data_path=../../data/train_data.mini.in
    eval_data_path=../../data/eval_data.mini.in
    epoch=1
    recall_k=50
    min_count=50
else
    train_data_path=../../data/train_data.in
    eval_data_path=../../data/eval_data.in
    epoch=1
    recall_k=10
    min_count=30
fi

lr=0.5
embedding_dim=128
train_ws=20
train_lower_ws=1
t=0.025
batch_size=64
num_sampled=10
hidden_units=""
prefetch_size=10000
max_train_steps=-1
save_summary_steps=100
save_checkpoints_secs=600
log_step_count_steps=1000
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
chief_lock=${MODEL_DIR}/chief.lock
max_distribute_train_steps=-1
train_nce_biases=1
shuffle_batch=1
predict_ws=20
sample_dropout=0.0
optimizer_type='sgd'  # 'ada', 'sgd', 'adadelta', 'adam'
tfrecord_file='../../data/train_data.tfrecord'
num_tfrecord_file=2
train_data_format='fasttext'  # 'tfrecord', 'fasttext'
tfrecord_map_num_parallel_calls=2
train_parallel_mode='train_op_parallel' # 'default', 'train_op_parallel'
num_train_op_parallel=4
use_batch_normalization=1
sgd_lr_decay_type='fasttext_decay'  # 'exponential_decay', 'fasttext_decay'
sgd_lr_decay_steps=100
sgd_lr_decay_rate=0.95
use_clip_gradients=1
clip_norm=5.0


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
    --num_sampled ${num_sampled} \
    --epoch ${epoch} \
    --hidden_units "${hidden_units}" \
    --model_dir ${MODEL_DIR} \
    --export_model_dir ${EXPORT_MODEL_DIR} \
    --prefetch_size ${prefetch_size} \
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
    --tfrecord_map_num_parallel_calls ${tfrecord_map_num_parallel_calls} \
    --train_parallel_mode ${train_parallel_mode} \
    --num_train_op_parallel ${num_train_op_parallel} \
    --use_batch_normalization ${use_batch_normalization} \
    --sgd_lr_decay_type ${sgd_lr_decay_type} \
    --sgd_lr_decay_steps ${sgd_lr_decay_steps} \
    --sgd_lr_decay_rate ${sgd_lr_decay_rate} \
    --use_clip_gradients ${use_clip_gradients} \
    --clip_norm ${clip_norm}
