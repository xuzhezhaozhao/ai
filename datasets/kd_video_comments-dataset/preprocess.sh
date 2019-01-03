#! /usr/bin/env bash

set -e

./grep_keywords.sh ./neg_dict.txt
./grep_pos.sh ./neg_dict.txt

mkdir -p data/preprocessed
python transform.py ./data/neg_uniq_shuf.txt ./data/neg_process.txt
python transform.py ./data/pos_uniq_shuf.txt ./data/pos_process.txt

cat ./data/neg_process.txt | sed 's/[ ][ ]*/ /g' | sed '/^$/d' > ./data/preprocessed/neg.txt
cat ./data/pos_process.txt | sed 's/[ ][ ]*/ /g' | sed '/^$/d' > ./data/preprocessed/pos.txt

cd ./data/preprocessed
head neg.txt -n 340000 > train-neg.txt
tail neg.txt -n 100000 > test-neg.txt

head pos.txt -n 3000000 > train-pos.txt
tail pos.txt -n 600000 > test-pos.txt
cd -
