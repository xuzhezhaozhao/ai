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

g++ -std=c++11 split_pos_neg.cpp -o data/split_pos_neg
./data/split_pos_neg \
    ${dict_file} \
    ./comments/kd_video_comments.csv \
    data/pos.txt \
    data/neg.txt
sed -i 's/^.*\t//g' data/pos.txt
sed -i 's/^.*\t//g' data/neg.txt

cat data/neg.txt | shuf > data/neg_shuf.txt
neg_lines=$(wc -l ./data/neg_shuf.txt | awk '{print $1}')
pos_lines=`echo "scale=2;${neg_lines}*5.0"|bc|awk '{print int($1)}'`
cat data/pos.txt | shuf | head -n ${pos_lines} > data/pos_shuf.txt
# cat data/pos.txt | shuf > data/pos_shuf.txt

python transform.py data/neg_shuf.txt data/neg_tokens.txt
python transform.py data/pos_shuf.txt data/pos_tokens.txt

sed -i 's/[ ][ ]*/ /g' data/neg_tokens.txt
sed -i 's/[ ][ ]*/ /g' data/pos_tokens.txt
sed -i '/^$/d' data/neg_tokens.txt
sed -i '/^$/d' data/pos_tokens.txt

mkdir -p data/nbsvm
mkdir -p data/fasttext
mkdir -p char-cnn/

# nbsvm
total_lines=$(wc -l data/neg_shuf.txt | awk '{print $1}')
train_lines=`echo "scale=2;${total_lines}*0.9"|bc|awk '{print int($1)}'`
test_lines=`echo "scale=2;${total_lines}*0.1"|bc|awk '{print int($1)}'`
head data/neg_tokens.txt -n ${train_lines} > data/nbsvm/train-neg.txt
tail data/neg_tokens.txt -n ${test_lines} > data/nbsvm/test-neg.txt

head data/neg_shuf.txt -n ${train_lines} > data/char-cnn/train-neg.txt.tmp
tail data/neg_shuf.txt -n ${test_lines} > data/char-cnn/test-neg.txt.tmp

total_lines=$(wc -l data/pos_shuf.txt | awk '{print $1}')
train_lines=`echo "scale=2;${total_lines}*0.9"|bc|awk '{print int($1)}'`
test_lines=`echo "scale=2;${total_lines}*0.1"|bc|awk '{print int($1)}'`
head data/pos_tokens.txt -n ${train_lines} > data/nbsvm/train-pos.txt
tail data/pos_tokens.txt -n ${test_lines} > data/nbsvm/test-pos.txt

head data/pos_shuf.txt -n ${train_lines} > data/char-cnn/train-pos.txt.tmp
tail data/pos_shuf.txt -n ${test_lines} > data/char-cnn/test-pos.txt.tmp

# fasttext
sed 's/^/__label__pos / ' data/nbsvm/train-pos.txt > data/fasttext/train-pos.tmp
sed 's/^/__label__pos / ' data/nbsvm/test-pos.txt > data/fasttext/test-pos.tmp
sed 's/^/__label__neg / ' data/nbsvm/train-neg.txt > data/fasttext/train-neg.tmp
sed 's/^/__label__neg / ' data/nbsvm/test-neg.txt > data/fasttext/test-neg.tmp

cat data/fasttext/train-*.tmp | shuf > data/fasttext/train.txt
cat data/fasttext/test-*.tmp | shuf > data/fasttext/test.txt
rm -rf data/fasttext/*.tmp

# char-cnn
sed -i 's/^/__label__pos / ' data/char-cnn/train-pos.txt.tmp
sed -i 's/^/__label__pos / ' data/char-cnn/test-pos.txt.tmp
sed -i 's/^/__label__neg / ' data/char-cnn/train-neg.txt.tmp
sed -i 's/^/__label__neg / ' data/char-cnn/test-neg.txt.tmp
cat data/char-cnn/train-*.tmp | shuf > data/char-cnn/train.txt
cat data/char-cnn/test-*.tmp | shuf > data/char-cnn/test.txt
rm -rf data/char-cnn/*.tmp
