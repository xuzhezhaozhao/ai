
import os

cats = []
dogs = []

train_dir = '../../../datasets/cat-dog-dataset/train/'
for filename in os.listdir(train_dir):
    tokens = filename.split('.')
    if tokens[0] == 'cat':
        cats.append(train_dir + filename)

    elif tokens[0] == 'dog':
        dogs.append(train_dir + filename)

    else:
        raise TypeError("Wrong format data.")

# split
split = int(0.8 * len(cats))
train_cats = cats[0:split]
eval_cats = cats[split:]

split = int(0.8 * len(dogs))
train_dogs = dogs[0:split]
eval_dogs = dogs[split:]

with open('train.txt', 'w') as f:
    for item in train_cats:
        f.write(item + ' ' + '0\n')

    for item in train_dogs:
        f.write(item + ' ' + '1\n')

with open('eval.txt', 'w') as f:
    for item in eval_cats:
        f.write(item + ' ' + '0\n')

    for item in eval_dogs:
        f.write(item + ' ' + '1\n')


with open('test.txt', 'w') as f:
    test_dir = '../../../datasets/cat-dog-dataset/test/'
    for filename in os.listdir(test_dir):
        tokens = filename.split('.')
        f.write(test_dir + filename)
        f.write('\n')
