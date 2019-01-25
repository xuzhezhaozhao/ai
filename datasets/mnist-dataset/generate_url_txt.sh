
datadir=data/train
find ${datadir} -type f | sed "s:^:`pwd`/:" | shuf > train.txt
