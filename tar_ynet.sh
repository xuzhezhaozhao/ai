tar cvzf ynet.tgz ynet/ \
    --exclude=ynet/model  \
    --exclude=ynet/*.pyc  \
    --exclude=ynet/data/data.in  \
    --exclude=ynet/data/data.in.noheader  \
    --exclude=ynet/data/data.in.noheader.bin  \
    --exclude=ynet/data/data.in.noheader.preprocessed  \
    --exclude=ynet/data/data.in.noheader.sorted  \
    --exclude=ynet/data/data.in.noheader.vec  \
    --exclude=ynet/data.bak
