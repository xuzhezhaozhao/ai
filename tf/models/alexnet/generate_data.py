
import os

with open('train.txt', 'w') as f:
    train_dir = '../../../datasets/cat-dog-dataset/train/'
    for filename in os.listdir(train_dir):
        tokens = filename.split('.')
        if tokens[0] == 'cat':
            f.write(train_dir + filename)
            f.write(' ')
            f.write('0\n')
        elif tokens[0] == 'dog':
            f.write(train_dir + filename)
            f.write(' ')
            f.write('1\n')
        else:
            raise TypeError("Wrong format data.")


with open('test.txt', 'w') as f:
    test_dir = '../../../datasets/cat-dog-dataset/test/'
    for filename in os.listdir(test_dir):
        tokens = filename.split('.')
        f.write(test_dir + filename)
        f.write('\n')
