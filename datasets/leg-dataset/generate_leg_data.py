import os

legs = []
ugcs = []
basedir = os.path.split(os.path.realpath(__file__))[0]

train_dir = os.path.join(basedir, 'train/')

for filename in os.listdir(train_dir):
    tokens = filename.split('_')
    if tokens[0] == 'leg':
        legs.append(os.path.join(train_dir, filename))
    elif tokens[0] == 'ugc':
        ugcs.append(os.path.join(train_dir, filename))
    else:
        raise TypeError("Wrong format data.")

# split train and validation
split = int(0.9 * len(legs))
train_legs = legs[0:split]
eval_legs = legs[split:]

split = int(0.9 * len(ugcs))
train_ugcs = ugcs[0:split]
eval_ugcs = ugcs[split:]

with open('train.txt', 'w') as f:
    for item in train_legs:
        f.write(item + ' ' + '0\n')

    for item in train_ugcs:
        f.write(item + ' ' + '1\n')

with open('validation.txt', 'w') as f:
    for item in eval_legs:
        f.write(item + ' ' + '0\n')

    for item in eval_ugcs:
        f.write(item + ' ' + '1\n')


with open('test.txt', 'w') as f:
    test_dir = os.path.join(basedir, 'test/')
    for filename in os.listdir(test_dir):
        f.write(os.path.join(test_dir, filename))
        f.write('\n')
