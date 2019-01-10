
import os

with open('train.txt', 'w') as f:
    for filename in os.listdir('./cat-dog-dataset/train/'):
        tokens = filename.split('.')
        if tokens[0] == 'cat':
            f.write('./cat-dog-dataset/train/' + filename)
            f.write(' ')
            f.write('0\n')
        elif tokens[0] == 'dog':
            f.write('./cat-dog-dataset/train/' + filename)
            f.write(' ')
            f.write('1\n')
        else:
            raise TypeError("Wrong format data.")


with open('test.txt', 'w') as f:
    for filename in os.listdir('./cat-dog-dataset/test/'):
        tokens = filename.split('.')
        f.write('./cat-dog-dataset/test/' + filename)
        f.write('\n')
