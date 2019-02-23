#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

mkdir -p data
mkdir -p data/fasttext
mkdir -p data/char-cnn
python transform.py ./thucnews data/thucnews_tokens.txt data/thucnews.txt
cat data/thucnews_tokens.txt | shuf > data/thucnews_tokens_shuf.txt

total_lines=$(wc -l data/thucnews_tokens_shuf.txt | awk '{print $1}')
train_lines=`echo "scale=2;${total_lines}*0.8"|bc|awk '{print int($1)}'`
test_lines=`echo "scale=2;${total_lines}*0.2"|bc|awk '{print int($1)}'`
head data/thucnews_tokens_shuf.txt -n ${train_lines} > data/fasttext/train.txt
tail data/thucnews_tokens_shuf.txt -n ${test_lines} > data/fasttext/test.txt
