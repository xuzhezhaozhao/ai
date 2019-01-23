
datadir=data
find ${datadir} -type f | sed "s:^:`pwd`/:" | shuf > train.txt
