#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import codecs
import pickle

BUFFER_SIZE = 1024 * 1024


def preprocess(filename, preprocessed_filename, min_count):
    with codecs.open(filename, 'rb', encoding='utf-8') as f:
        total = 0
        chars_count = {}

        while True:
            data = f.read(BUFFER_SIZE)
            read_size = len(data)
            if read_size == 0:
                break
            total += read_size

            for char in data:
                if char not in chars_count:
                    chars_count[char] = 0
                chars_count[char] += 1

        tf.logging.info("[preprocess] total chars = {}".format(total))
        tf.logging.info("[preprocess] unique chars = {}"
                        .format(len(chars_count)))
        vocab = [c for c in chars_count.keys() if chars_count[c] >= min_count]
        vocab.insert(0, '<unk>')

        tf.logging.info("[preprocess] unique chars count large than {} = {}"
                        .format(min_count, len(vocab) - 1))

        dump_dict = {}
        dump_dict['vocab'] = vocab
        dump_dict['total_chars'] = total

        with open(preprocessed_filename, 'w') as f:
            pickle.dump(dump_dict, f)


def load_preprocessed(filename):
    with open(filename, 'rb') as f:
        load_dict = pickle.load(f)

    return load_dict


def text_to_int(text, opts):
    load_dict = load_preprocessed(opts.preprocessed_filename)
    vocab = load_dict['vocab']
    char2idx = {u: i for i, u in enumerate(vocab)}
    text_as_int = np.array([char2idx[c] if c in vocab else 0 for c in text])
    return text_as_int


def idx_to_text(idxs, opts):
    load_dict = load_preprocessed(opts.preprocessed_filename)
    vocab = load_dict['vocab']
    idx2char = np.array(vocab)
    text = idx2char[idxs]
    return text


def parse_txt(filename, opts):
    with codecs.open(filename, 'rb', encoding='utf-8') as f:
        data = f.read()

    return text_to_int(data, opts)


def build_train_input_fn_v2(opts, filename):
    def train_input_fn():
        data = parse_txt(filename, opts)
        chunk_size = opts.batch_size * opts.seq_length
        n_batches = int((len(data) - opts.batch_size) / chunk_size)
        data = data[:chunk_size * n_batches + opts.batch_size]
        data = data.reshape((opts.batch_size, -1))
        while True:
            np.random.shuffle(data)
            for n in range(0, data.shape[1]-opts.seq_length, opts.seq_length):
                x = data[:, n:n + opts.seq_length]
                y = np.zeros_like(x)
                y[:, :-1] = x[:, 1:]
                y[:, -1] = data[:, n + opts.seq_length]
                yield x, y

    return train_input_fn
