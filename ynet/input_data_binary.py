#! /usr/bin/env python
# -*-coding:utf-8 -*-

import tensorflow as tf
import struct
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base


class DataSet(object):
    def __init__(self, watched_videos, predicts):
        self._watched_videos = watched_videos
        self._predicts = predicts
        self._epochs_completed = 0
        self._index_in_epoch = 0

        @property
        def watched_videos(self):
            return self._watched_videos

        @property
        def predicts(self):
            return self._predicts

        @property
        def num_examples(self):
            return self._num_examples

        @property
        def epochs_completed(self):
            return self._epochs_completed

        def next_batch(self, batch_size, shuffle=True):
            start = self._index_in_epoch
            # Shuffle for the first epoch
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = np.arange(self._num_examples)
                np.random.shuffle(perm0)
                self._images = self.images[perm0]
                self._labels = self.labels[perm0]
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start
                watched_videos_rest_part = \
                    self._watched_videos[start:self._num_examples]
                predicts_rest_part = self._predicts[start:self._num_examples]
                # Shuffle the data
                if shuffle:
                    perm = np.arange(self._num_examples)
                    np.random.shuffle(perm)
                    self._watched_videos = self.watched_videos[perm]
                    self._predicts = self.predicts[perm]
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch
                watched_videos_new_part = self._watched_videos[start:end]
                predicts_new_part = self._predicts[start:end]
                return \
                    np.concatenate((watched_videos_rest_part,
                                    watched_videos_new_part), axis=0), \
                    np.concatenate((predicts_rest_part,
                                    predicts_new_part), axis=0)
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                return self._watched_videos[start:end], \
                    self._predicts[start:end]


def load_video_embeddings_from_binary(binaryfile, dictfile):
    num = 0
    dim = 0
    with open(binaryfile, 'rb') as fbinary:
        databytes = fbinary.read()
        num = struct.unpack('<i', databytes[0:4])[0]
        dim = struct.unpack('<i', databytes[4:8])[0]
        embeddings = tf.decode_raw(databytes[8:], tf.float32)
        embeddings = tf.reshape(embeddings, shape=(num, dim))

    return embeddings, num, dim


def read_data_sets_from_binary(train_watched_file,
                               train_predicts_file,
                               validation_watched_file,
                               validation_predicts_file,
                               test_watched_file,
                               test_predicts_file,
                               watched_size):
    if train_watched_file:
        with open(train_watched_file, 'rb') as f:
            databytes = f.read()
            train_watched = np.fromfile(databytes, tf.int32)
            total = train_watched.shape[0]
            assert (total % watched_size == 0)
            nlines = total / watched_size
            train_watched = train_watched.reshape(nlines, watched_size)

    if train_predicts_file:
        with open(train_predicts_file, 'rb') as f:
            databytes = f.read()
            train_predicts = np.fromfile(databytes, tf.int32)
            total = train_predicts.shape[0]
            assert (total % watched_size == 0)
            nlines = total / watched_size
            train_predicts = train_predicts.reshape(nlines, watched_size)

    if validation_watched_file:
        with open(validation_watched_file, 'rb') as f:
            databytes = f.read()
            validation_watched = np.fromfile(databytes, tf.int32)
            total = validation_watched.shape[0]
            assert (total % watched_size == 0)
            nlines = total / watched_size
            validation_watched = validation_watched.reshape(nlines,
                                                            watched_size)
    if validation_predicts_file:
        with open(validation_predicts_file, 'rb') as f:
            databytes = f.read()
            validation_predicts = np.fromfile(databytes, np.int32)
            total = validation_predicts.shape[0]
            assert (total % watched_size == 0)
            nlines = total / watched_size
            validation_predicts = validation_predicts.reshape(nlines,
                                                              watched_size)

    if test_watched_file:
        with open(test_watched_file, 'rb') as f:
            databytes = f.read()
            test_watched = np.fromfile(databytes, tf.int32)
            total = test_watched.shape[0]
            assert (total % watched_size == 0)
            nlines = total / watched_size
            test_watched = test_watched.reshape(nlines, watched_size)
    if test_predicts_file:
        with open(test_predicts_file, 'rb') as f:
            databytes = f.read()
            test_predicts = np.fromfile(databytes, tf.int32)
            total = test_predicts.shape[0]
            assert (total % watched_size == 0)
            nlines = total / watched_size
            test_predicts = test_predicts.reshape(nlines, watched_size)

    train = DataSet(train_watched, train_predicts)
    validation = DataSet(validation_watched, validation_predicts)
    test = DataSet(test_watched, test_predicts)

    return base.Datasets(train=train, validation=validation, test=test)
