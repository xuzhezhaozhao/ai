
import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--channel_num', default=0, type=int, help='')
parser.add_argument('--filter_num', default=0, type=int, help='')
parser.add_argument('--layer', default='', type=str, help='')
parser.add_argument('--show_type', default='', type=str, help='')


CONV_LAYERS = ('conv1', 'conv2', 'conv3', 'conv4', 'conv5')


def show_all_layers(args):
    data_dict = np.load('./bvlc_alexnet.npy').item()

    hspace = 0.6
    plt.subplots_adjust(hspace=hspace)
    for layer_index, layer in enumerate(CONV_LAYERS):
        flt = data_dict[layer][0]
        print("filter '{}', shape '{}'".format(layer, flt.shape))
        plt.subplot(2, 3, layer_index+1)
        im = plt.imshow(flt[:, :, args.channel_num, args.filter_num],
                        cmap=plt.cm.gray)
        plt.colorbar(im)
        plt.title(layer + ':' + str(args.channel_num), y=1.0)
    plt.show()


def show_single_layer(args):
    data_dict = np.load('./bvlc_alexnet.npy').item()
    hspace = 0.6
    plt.subplots_adjust(hspace=hspace)
    flt = data_dict[args.layer][0]
    num_filters = flt.shape[3]
    hspace = 0.6
    plt.subplots_adjust(hspace=hspace)
    for index in range(num_filters):
        if index >= 64:
            break
        plt.subplot(8, 8, index+1)
        im = plt.imshow(flt[:, :, args.channel_num, index], cmap=plt.cm.gray)
        plt.colorbar(im)
        plt.title(args.layer + ':' + str(index), y=1.0)
    plt.show()


def main(argv):
    args = parser.parse_args(argv[1:])
    if args.show_type == 'all':
        show_all_layers(args)
    elif args.show_type == 'single':
        show_single_layer(args)
    else:
        raise ValueError("Not surpported show type.")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
