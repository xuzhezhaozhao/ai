
sorted_file=mini.in
min_count=50
preprocessed=mini.in.preprocessed.2
./build/src/preprocess \
    -raw_input=${sorted_file} \
    -with_header=false \
    -only_video=true \
    -interval=1000000 \
    -output_user_watched_file=${preprocessed} \
    -output_user_watched_ratio_file=${preprocessed}.watched_ratio \
    -output_video_play_ratio_file=${preprocessed}.play_raito \
    -user_min_watched=1 \
    -user_max_watched=1024 \
    -user_abnormal_watched_thr=2048 \
    -supress_hot_arg1=8 \
    -supress_hot_arg2=3 \
    -user_effective_watched_time_thr=5 \
    -user_effective_watched_ratio_thr=0.05 \
    -min_count=${min_count}
