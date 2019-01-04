#! /usr/bin/env bash

set -e

if [ $# == 0 ] ; then
    dict_file=./dict/neg_dict.txt
elif [ $# == 1 ] ; then
    dict_file=$1
else
    echo "Usage: $0 <neg_dic>"
    exit -1
fi

rm -rf data
mkdir data
python split_pos_neg.py ${dict_file} ./data/pos.txt ./data/neg.txt
sed -i 's/^.*\t//g' data/pos.txt
sed -i 's/^.*\t//g' data/neg.txt

cat data/neg.txt | sort | uniq | shuf > ./data/neg_uniq_shuf.txt
neg_lines=$(wc -l ./data/neg_uniq_shuf.txt | awk '{print $1}')
pos_lines=`echo "scale=2;${neg_lines}*5.0"|bc|awk '{print int($1)}'`
cat data/pos.txt | sort | uniq | shuf | head -n ${pos_lines} > ./data/pos_uniq_shuf.txt

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
