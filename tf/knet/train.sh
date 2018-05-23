
rm -rf model_dir
python train.py \
    --train_data_path ../ops/ftlib/train_data.in \
    --lr 0.5 \
    --dim 100 \
    --maxn 0 \
    --minn 0 \
    --word_ngrams 1 \
    --bucket 2000000 \
    --ws 20 \
    --min_count 50 \
    --t 0.0001 \
    --verbose 1 \
    --min_count_label 1 \
    --label "__label__" \
    --batch_size 258 \
    --num_sampled 10 \
    --epoch 100 \
    --hidden_units "128,64" \
    --model_dir "model_dir" \
    --export_model_dir "export_model_dir" \
    --nclasses 10000 \
    --prefetch_size 100000 \
    --save_summary_steps 100 \
    --save_checkpoints_secs 600 \
    --keep_checkpoint_max 3 \
    --log_step_count_steps 100
