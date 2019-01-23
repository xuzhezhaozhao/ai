
datadir=lsun_bedroom
find ${datadir} | sed "s:^:`pwd`/:" | shuf > train.txt
