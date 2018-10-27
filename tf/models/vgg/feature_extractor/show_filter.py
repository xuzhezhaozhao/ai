
import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--channel_num', default=0, type=int, help='')
parser.add_argument('--filter_num', default=0, type=int, help='')
parser.add_argument('--layer', default='', type=str, help='')
parser.add_argument('--show_type', default='', type=str, help='')


CONV_LAYERS = (
    ['conv1_1', 'conv1_2'],
    ['conv2_1', 'conv2_2'],
    ['conv3_1', 'conv3_2', 'conv3_3', 'conv3_4'],
    ['conv4_1', 'conv4_2', 'conv4_3', 'conv4_4'],
    ['conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
)


def show_all_layers(args):
    data_dict = np.load('./vgg19.npy').item()

    hspace = 0.6
    plt.subplots_adjust(hspace=hspace)
    for layer_index, layer in enumerate(CONV_LAYERS):
        for sublayer_index, sublayer in enumerate(layer):
            flt = data_dict[sublayer][0]
            print("filter '{}', shape '{}'".format(sublayer, flt.shape))
            plt.subplot(5, 4, (layer_index*4 + sublayer_index)+1)
            im = plt.imshow(flt[:, :, args.channel_num, args.filter_num],
                            cmap=plt.cm.gray)
            plt.colorbar(im)
            plt.title(sublayer + ':' + str(args.channel_num), y=1.0)
    plt.show()


def show_single_layer(args):
    data_dict = np.load('./vgg19.npy').item()
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
