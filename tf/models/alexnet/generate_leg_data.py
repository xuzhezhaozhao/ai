
import os

legs = []
ugcs = []

train_dir = '../../../datasets/leg-dataset/train/'
for filename in os.listdir(train_dir):
    tokens = filename.split('_')
    if tokens[0] == 'leg':
        legs.append(train_dir + filename)

    elif tokens[0] == 'ugc':
        ugcs.append(train_dir + filename)

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
    test_dir = '../../../datasets/leg-dataset/test/'
    for filename in os.listdir(test_dir):
        tokens = filename.split('_')
        f.write(test_dir + filename)
        f.write('\n')
