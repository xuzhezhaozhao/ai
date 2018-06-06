MODEL_DIR=`pwd`/model_dir
EXPORT_MODEL_DIR=`pwd`/export_model_dir
train_data_path=../../data/train_data.in
eval_data_path=../../data/eval_data.in
lr=0.25
dim=100
ws=20
min_count=50
batch_size=64
num_sampled=10
epoch=5
hidden_units="64"
prefetch_size=10000

max_train_steps=-1

save_summary_steps=100
save_checkpoints_secs=600
log_step_count_steps=1000

recall_k=10
dict_dir=`pwd`/dict_dir
use_saved_dict=0

use_profile_hook=0
profile_steps=100

root_ops_path=lib/
remove_model_dir=1

optimize_level=2

receive_ws=100

use_subset=1

python train.py \
    --train_data_path ${train_data_path} \
    --eval_data_path ${eval_data_path} \
    --lr ${lr} \
    --dim ${dim} \
    --maxn 0 \
    --minn 0 \
    --word_ngrams 1 \
    --bucket 2000000 \
    --ws ${ws} \
    --min_count ${min_count} \
    --t 0.0001 \
    --verbose 2 \
    --min_count_label 50 \
    --label "__label__" \
    --batch_size ${batch_size} \
    --num_sampled ${num_sampled} \
    --epoch ${epoch} \
    --hidden_units ${hidden_units} \
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
    --use_subset ${use_subset}
