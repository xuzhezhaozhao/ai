#! /usr/bin/env bash

set -e

# dict_file=./neg_dict.txt
dict_file=$1

./grep_neg.sh ${dict_file}
./grep_pos.sh ${dict_file}

mkdir -p data/preprocessed
python transform.py ./data/neg_uniq_shuf.txt ./data/neg_process.txt
python transform.py ./data/pos_uniq_shuf.txt ./data/pos_process.txt

cat ./data/neg_process.txt | sed 's/[ ][ ]*/ /g' | sed '/^$/d' > ./data/preprocessed/neg.txt
cat ./data/pos_process.txt | sed 's/[ ][ ]*/ /g' | sed '/^$/d' > ./data/preprocessed/pos.txt

cd ./data/preprocessed
total_lines=$(wc -l neg.txt | awk '{print $1}')
train_lines=`echo "scale=2;${total_lines}*0.8"|bc|awk '{print int($1)}'`
test_lines=`echo "scale=2;${total_lines}*0.2"|bc|awk '{print int($1)}'`
head neg.txt -n ${train_lines} > train-neg.txt
tail neg.txt -n ${test_lines} > test-neg.txt

total_lines=$(wc -l pos.txt | awk '{print $1}')
train_lines=`echo "scale=2;${total_lines}*0.8"|bc|awk '{print int($1)}'`
test_lines=`echo "scale=2;${total_lines}*0.2"|bc|awk '{print int($1)}'`
head pos.txt -n ${total_lines} > train-pos.txt
tail pos.txt -n ${test_lines} > test-pos.txt

mkdir -p fasttext
sed 's/^/__label__pos / ' train-pos.txt > fasttext/train-pos.txt
sed 's/^/__label__pos / ' test-pos.txt > fasttext/test-pos.txt
sed 's/^/__label__neg / ' train-neg.txt > fasttext/train-neg.txt
sed 's/^/__label__neg / ' test-neg.txt > fasttext/test-neg.txt

cat fasttext/train-* > fasttext/train.txt.tmp
cat fasttext/test-* > fasttext/test.txt.tmp
shuf fasttext/train.txt.tmp -o fasttext/train.txt
shuf fasttext/test.txt.tmp -o fasttext/test.txt
cd ../../..

python filter_rowkey.py
