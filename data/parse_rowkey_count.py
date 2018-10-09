

D = dict()
for line in open('krank_train_data.in', 'r'):
    tokens = line.split('\t')[1].split(' ')
    for token in tokens:
        rowkey = token.split(':')[0]
        if rowkey in D:
            D[rowkey] += 1
        else:
            D[rowkey] = 1


with open('rowkey_count.csv', 'w') as f:
    for k in D:
        f.write(k)
        f.write('\t')
        f.write(str(D[k]))
        f.write('\n')
