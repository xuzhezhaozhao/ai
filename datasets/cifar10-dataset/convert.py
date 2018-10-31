
"""
code from: https://www.quora.com/How-can-l-visualize-cifar-10-data-RGB-using-python-matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import misc


cifar10_dir = './cifar-10-batches-py'
data_batchs = ['data_batch_1', 'data_batch_2', 'data_batch_3',
               'data_batch_4', 'data_batch_5']
test_batch = 'test_batch'


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def get_data(filename):
    absFile = os.path.abspath(filename)
    dict = unpickle(absFile)
    for key in dict.keys():
            print(key)
    print("Unpacking {}".format(dict[b'batch_label']))
    X = np.asarray(dict[b'data']).astype("uint8")
    Y = np.asarray(dict[b'labels'])
    names = np.asarray(dict[b'filenames'])
    return X, Y, names


def visualize_image(X, Y, names, id):
    rgb = X[id, :]
    print(rgb.shape)
    img = rgb.reshape(3, 32, 32).transpose([1, 2, 0])
    print(img.shape)
    plt.imshow(img)
    plt.title(names[id])
    plt.show()


def convert_to_png(filename, start_id, output_dir):
    X, Y, _ = get_data(filename)

    for row in range(X.shape[0]):
        rgb = X[row, :]
        img = rgb.reshape(3, 32, 32).transpose([1, 2, 0])
        save_name = os.path.join(
            output_dir, str(start_id) + '.' + str(Y[row]) + '.png')
        start_id += 1
        misc.toimage(img, cmin=0, cmax=255).save(save_name)

    return start_id


def main():
    start_id = 0
    for train_batch in data_batchs:
        filename = os.path.join(cifar10_dir, train_batch)
        start_id = convert_to_png(filename, start_id, 'train')

    start_id = 0
    filename = os.path.join(cifar10_dir, test_batch)
    start_id = convert_to_png(filename, start_id, 'test')


if __name__ == '__main__':
    main()
