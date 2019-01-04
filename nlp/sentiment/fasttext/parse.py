
D = dict()
for line in open('./result.txt'):
    rowkey, sentiment = line.strip().split(' ')
    if rowkey not in D:
        D[rowkey] = [0, 0]

    if sentiment == '__label__pos':
        D[rowkey][0] += 1
    if sentiment == '__label__neg':
        D[rowkey][1] += 1

with open('sentiment.txt', 'w') as f:
    for rowkey in D:
        f.write(rowkey)
        f.write(' ')
        f.write(str(D[rowkey][0]))
        f.write(' ')
        f.write(str(D[rowkey][1]))
        f.write(' ')
        f.write(str(1.0*D[rowkey][1] / (D[rowkey][0] + D[rowkey][1])))
        f.write('\n')
