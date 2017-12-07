#! /usr/bin/env python
# -*-coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base
import numpy as np
import random
import struct

# dict for video key to embeddings index
D = dict()


class DataSet(object):
    def __init__(self, records, watched_size):
        """Construct a DataSet.
        Args:
            records: Python list of list.
        """
        self._records = records
        self._num_examples = len(records)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._watched_size = watched_size

    @property
    def records(self):
        return self._records

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._records = [self.records[i] for i in perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            records_rest_part, predicts_rest_part = \
                generate_batch(self._records[start:self._num_examples],
                               self._watched_size)
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._records = [self.records[i] for i in perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            records_new_part, predicts_new_part = \
                generate_batch(self._records[start:end],
                               self._watched_size)
            return \
                np.concatenate((records_rest_part, records_new_part),
                               axis=0), \
                np.concatenate((predicts_rest_part, predicts_new_part),
                               axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            if self._index_in_epoch == self._num_examples:
                self._index_in_epoch = 0
                self._epochs_completed += 1
            return generate_batch(self._records[start:end],
                                  self._watched_size)


def generate_batch(records, watched_size):
    watched = []
    predicts = []
    for record in records:
        assert (len(record) > watched_size), (
            'watched record size {} should larger than --watched_size {}'.
            format(len(record), watched_size))
        max_start = len(record) - watched_size - 1
        start = random.randint(0, max_start)
        watched.append(record[start:start + watched_size])
        predicts.append(record[start + watched_size:start + watched_size + 1])
    return watched, predicts


def load_video_embeddings(filename):
    """ Load pretrained video embeddings from file
    Return:
        embeddings: A shape (num, dim) Tensor. Pretrained video embeddings.
        num: A int. Number of videos.
        dim: A int. Embedding dimension.
        D: A python dict. video key - embedding index in embeddings.
    """
    global D

    num = 0
    dim = 0

    with open(filename, "r") as f:
        line = f.readline().strip()
        tokens = line.split(' ')
        num, dim = map(int, tokens)

    embeddings = np.genfromtxt(filename, dtype='float32', delimiter=' ',
                               skip_header=1, usecols=range(1, dim + 1))
    embeddings = tf.convert_to_tensor(embeddings, dtype='float32')

    keys = np.genfromtxt(filename, dtype='string', delimiter=' ',
                         skip_header=1, usecols=0)
    D = {key: index for index, key in enumerate(keys)}
    return embeddings, num, dim


def load_video_embeddings_from_binary(binaryfile, dictfile):
    global D

    num = 0
    dim = 0
    with open(binaryfile, 'rb') as fbinary:
        databytes = fbinary.read()
        num = struct.unpack('<i', databytes[0:4])[0]
        dim = struct.unpack('<i', databytes[4:8])[0]
        embeddings = tf.decode_raw(databytes[8:], tf.float32)
        embeddings = tf.reshape(embeddings, shape=(num, dim))

    for index, line in enumerate(open(dictfile, 'r')):
        D[line.strip()] = index

    return embeddings, num, dim


def read_data_sets(train_file, validation_file, test_file, watched_size):
    global D

    train_records = []
    for line in open(train_file):
        items = line.strip().split(' ')[1:]
        record = [D[k] for k in items]
        train_records.append(record)

    validation_records = []
    for line in open(validation_file):
        items = line.split(' ')[1:]
        record = [D[k] for k in items]
        validation_records.append(record)

    test_records = []
    for line in open(test_file):
        items = line.split(' ')[1:]
        record = [D[k] for k in items]
        test_records.append(record)

    train = DataSet(train_records, watched_size)
    validation = DataSet(validation_records, watched_size)
    test = DataSet(test_records, watched_size)

    return base.Datasets(train=train, validation=validation, test=test)
