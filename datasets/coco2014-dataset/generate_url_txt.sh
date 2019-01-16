
datadir=data/train2014/
ls ${datadir} | sed "s:^:`pwd`/${datadir}:" | shuf > train.txt
