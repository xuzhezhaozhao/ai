#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

if [ $# == 0 ] ; then
    dict_file=dict/neg_dict.txt
elif [ $# == 1 ] ; then
    dict_file=$1
else
    echo "Usage: $0 <neg_dic>"
    exit -1
fi

rm -rf data
mkdir data
python split_pos_neg.py ${dict_file} data/pos.txt data/neg.txt
sed -i 's/^.*\t//g' data/pos.txt
sed -i 's/^.*\t//g' data/neg.txt

cat data/neg.txt | shuf > data/neg_shuf.txt
neg_lines=$(wc -l ./data/neg_shuf.txt | awk '{print $1}')
pos_lines=`echo "scale=2;${neg_lines}*5.0"|bc|awk '{print int($1)}'`
cat data/pos.txt | shuf | head -n ${pos_lines} > data/pos_shuf.txt

python transform.py data/neg_shuf.txt data/neg_tokens.txt
python transform.py data/pos_shuf.txt data/pos_tokens.txt

sed -i 's/[ ][ ]*/ /g' data/neg_tokens.txt
sed -i 's/[ ][ ]*/ /g' data/pos_tokens.txt
sed -i '/^$/d' data/neg_tokens.txt
sed -i '/^$/d' data/pos_tokens.txt

mkdir -p data/nbsvm
total_lines=$(wc -l data/neg.txt | awk '{print $1}')
train_lines=`echo "scale=2;${total_lines}*0.9"|bc|awk '{print int($1)}'`
test_lines=`echo "scale=2;${total_lines}*0.1"|bc|awk '{print int($1)}'`
head data/neg_tokens.txt -n ${train_lines} > data/nbsvm/train-neg.txt
tail data/neg_tokens.txt -n ${test_lines} > data/nbsvm/test-neg.txt

total_lines=$(wc -l data/pos.txt | awk '{print $1}')
train_lines=`echo "scale=2;${total_lines}*0.9"|bc|awk '{print int($1)}'`
test_lines=`echo "scale=2;${total_lines}*0.1"|bc|awk '{print int($1)}'`
head data/pos_tokens.txt -n ${total_lines} > data/nbsvm/train-pos.txt
tail data/pos_tokens.txt -n ${test_lines} > data/nbsvm/test-pos.txt

mkdir -p data/fasttext
sed 's/^/__label__pos / ' data/nbsvm/train-pos.txt > data/fasttext/train-pos.tmp
sed 's/^/__label__pos / ' data/nbsvm/test-pos.txt > data/fasttext/test-pos.tmp
sed 's/^/__label__neg / ' data/nbsvm/train-neg.txt > data/fasttext/train-neg.tmp
sed 's/^/__label__neg / ' data/nbsvm/test-neg.txt > data/fasttext/test-neg.tmp

cat data/fasttext/train-*.tmp | shuf > data/fasttext/train.txt
cat data/fasttext/test-*.tmp | shuf > data/fasttext/test.txt
rm -rf data/fasttext/*.tmp
