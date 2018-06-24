
cd ./ops/fasttext/
bash ./compile_writer.sh
cd -

tfrecord_file='../../data/train_data.tfrecord'
ws=20
min_count=30
t=0.01
ntargets=1
sample_dropout=0.0
train_data_path='../../data/train_data.in'
dict_dir='dict_dir'
threads=2
is_delete=1
./ops/fasttext/tfrecord_writer \
    --tfrecord_file ${tfrecord_file} \
    --ws ${ws} \
    --min_count ${min_count} \
    -t ${t} \
    --ntargets ${ntargets} \
    --sample_dropout ${sample_dropout} \
    --train_data_path ${train_data_path} \
    --dict_dir ${dict_dir} \
    --threads ${threads} \
    --is_delete ${is_delete}
