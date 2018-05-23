
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
    --batch_size 64 \
    --num_sampled 10 \
    --epoch 20 \
    --hidden_units "128,64" \
    --model_dir "model_dir" \
    --export_model_dir "export_model_dir"
