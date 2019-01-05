
mkdir -p model
datadir=../../../datasets/kd_video_comments-dataset/data/preprocessed/fasttext
../../../submodules/fastText/fasttext \
    supervised \
    -input ${datadir}/train.txt \
    -output model/model \
    -dim 50 \
    -lr 0.025 \
    -wordNgrams 2 \
    -minCount 10 \
    -bucket 500000 \
    -epoch 15 \
    -thread 7 \
    -minn 0 \
    -maxn 0

../../../submodules/fastText/fasttext \
    test \
    model/model.bin \
    ${datadir}/test.txt

../../../submodules/fastText/fasttext \
    predict \
    model/model.bin \
    ${datadir}/test.txt > preidct.txt

python check_error.py preidct.txt ${datadir}/test.txt > error.txt
