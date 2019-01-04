
datadir=../../../datasets/kd_video_comments-dataset/data/preprocessed/fasttext
../../../submodules/fastText/fasttext \
    supervised \
    -input ${datadir}/train.txt \
    -output model/model \
    -dim 50 \
    -lr 0.025 \
    -wordNgrams 3 \
    -minCount 5 \
    -bucket 10000000 \
    -epoch 20 \
    -thread 7


mkdir -p model
../../../submodules/fastText/fasttext \
    test \
    model/model.bin \
    ${datadir}/test.txt
