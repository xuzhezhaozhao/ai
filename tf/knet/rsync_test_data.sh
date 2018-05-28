
rsync -azvcP rsync://user_00@100.115.170.92:49020/data/kd_video_hbcf_offline/hbcf/data1/data.in.preprocessed ./

echo "shuf ..."
shuf data.in.preprocessed -o data.in.shuf
total_lines=$(wc -l data.in.shuf | awk '{print $1}')
eval_lines=500000
train_lines=$((total_lines-eval_lines))

echo "generate train_data ..."
head data.in.shuf -n ${train_lines} > train_data.in

echo "generate eval_data ..."
tail data.in.shuf -n ${eval_lines} > eval_data.in
