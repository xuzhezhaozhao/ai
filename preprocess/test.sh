
sorted_file=mini.in
min_count=50
preprocessed=mini.in.preprocessed.ban
./build/src/preprocess \
    -raw_input=${sorted_file} \
    -with_header=false \
    -only_video=false \
    -interval=1000000 \
    -output_user_watched_file=${preprocessed} \
    -output_user_watched_ratio_file=${preprocessed}.watched_ratio \
    -output_video_play_ratio_file=${preprocessed}.play_raito \
    -user_min_watched=1 \
    -user_max_watched=1024 \
    -user_abnormal_watched_thr=2048 \
    -supress_hot_arg1=10 \
    -supress_hot_arg2=3 \
    -user_effective_watched_time_thr=5 \
    -user_effective_watched_ratio_thr=0.05 \
    -min_count=${min_count} \
    -ban_algo_watched_ratio_thr=0.8 \
    -ban_algo_ids='3323,3321,3313,3312,3311,3310,3309,3308,3307,3306,3305,3304,3303,3302,3301' \
    -output_video_dict_file=${preprocessed}.video_dict \
    -output_article_dict_file=${preprocessed}.article_dict \
    -output_video_click_file=${preprocessed}.video_click
