#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import codecs
import pickle

BUFFER_SIZE = 1024*1024


def preprocess(filename, preprocessed_filename):
    with codecs.open(filename, 'rb', encoding='utf-8') as f:
        total = 0
        chars_dict = {}

        while True:
            data = f.read(BUFFER_SIZE)
            read_size = len(data)
            if read_size == 0:
                break
            total += read_size

            for char in data:
                if char not in chars_dict:
                    chars_dict[char] = 0
                chars_dict[char] += 1

        tf.logging.info("[preprocess] total chars = {}".format(total))
        tf.logging.info("[preprocess] unique chars = {}"
                        .format(len(chars_dict)))

        vocab = chars_dict.keys()
        vocab.insert(0, '#UNK#')  # UNK

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


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def build_train_input_fn(opts, filename):
    def train_input_fn():
        text_as_int = parse_txt(filename, opts)
        text_as_int = tf.data.Dataset.from_tensor_slices(text_as_int)
        chunks = text_as_int.batch(opts.seq_length+1, drop_remainder=True)
        dataset = chunks.map(split_input_target,
                             num_parallel_calls=opts.map_num_parallel_calls)
        if opts.shuffle_batch:
            dataset = dataset.shuffle(opts.shuffle_size)
        dataset = dataset.batch(opts.batch_size, drop_remainder=True)
        dataset = dataset.repeat(opts.epoch)

        return dataset

    return train_input_fn
