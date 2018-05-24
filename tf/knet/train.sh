MODEL_DIR=model_dir
EXPORT_MODEL_DIR=export_model_dir
train_data_path=../ops/ftlib/train_data.in
lr=0.5
dim=100
ws=20
min_count=50
batch_size=64
num_sampled=10
epoch=10
hidden_units="128,64"
nclasses=10000
prefetch_size=10000

save_summary_steps=100
save_checkpoints_secs=600
log_step_count_steps=100

rm -rf ${MODEL_DIR}
python train.py \
    --train_data_path ${train_data_path} \
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
    --nclasses ${nclasses} \
    --prefetch_size ${prefetch_size} \
    --save_summary_steps ${save_summary_steps} \
    --save_checkpoints_secs ${save_checkpoints_secs} \
    --keep_checkpoint_max 2 \
    --log_step_count_steps ${log_step_count_steps}
