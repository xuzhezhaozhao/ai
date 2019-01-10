
import numpy as np

files = ['train_data.in', 'eval_data.in']

cnt = 0
for filename in files:
    with open(filename, 'r') as fin, open(filename + '.new', 'w') as fout:
        for line in fin:
            line = line.strip()
            fout.write('__label__{}'.format(cnt))
            fout.write('\t')
            fout.write(line)
            fout.write('\n')

            cnt += 1

with open("user_features.tsv", 'w') as fuser:
    for i in range(cnt):
        fuser.write(str(i))
        fuser.write('\t')
        fuser.write(str(np.random.randint(10)))
        fuser.write('\t')
        fuser.write(str(np.random.randint(3)))
        fuser.write('\n')
