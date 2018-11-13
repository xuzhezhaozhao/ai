#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import os
import time


data_path = './data/tinyshakespeare/input.txt'
checkpoint_dir = './model_dir'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

text = open(data_path).read()
print ('Length of text: {} characters'.format(len(text)))

vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))

# Creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

for char, _ in zip(char2idx, range(20)):
    print('{:6s} ---> {:4d}'.format(repr(char), char2idx[char]))

# The maximum length sentence we want for a single input in characters
seq_length = 100

# Create training examples / targets
chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(
    seq_length + 1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = chunks.map(split_input_target)


# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(Model, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        if tf.test.is_gpu_available():
            self.gru = tf.keras.layers.CuDNNGRU(
                self.units,
                return_sequences=True,
                recurrent_initializer='glorot_uniform',
                stateful=True)
        else:
            self.gru = tf.keras.layers.GRU(
                self.units,
                return_sequences=True,
                recurrent_activation='sigmoid',
                recurrent_initializer='glorot_uniform',
                stateful=True)

        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        embedding = self.embedding(x)

        # output at every time step
        # output shape == (batch_size, seq_length, hidden_size)
        output = self.gru(embedding)

        # The dense layer will output predictions for every
        # time_steps(seq_length) output shape after the dense
        # layer == (seq_length * batch_size, vocab_size)
        prediction = self.fc(output)

        # states will be used to pass at every step to the model while training
        return prediction


# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
units = 1024

model = Model(vocab_size, embedding_dim, units)


# Using adam optimizer with default arguments
optimizer = tf.train.AdamOptimizer()


# Using sparse_softmax_cross_entropy so that we don't have to create one-hot
# vectors
def loss_function(real, preds):
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)


model.build(tf.TensorShape([BATCH_SIZE, seq_length]))
model.summary()

it = dataset.make_initializable_iterator()
inp, target = it.get_next()

predictions = model(inp)
loss_tensor = loss_function(target, predictions)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss_tensor)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(it.initializer)

    for step in xrange(200):
        _, loss = sess.run([train_op, loss_tensor])
        print("loss = {}".format(loss))

    model.save_weights(checkpoint_prefix)


# restore
model = Model(vocab_size, embedding_dim, units)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

# Evaluation step (generating text using the learned model)

# Number of characters to generate
num_generate = 1000

# You can change the start string to experiment
start_string = 'Q'

# Converting our start string to numbers (vectorizing)
input_eval = [char2idx[s] for s in start_string]
input_eval = tf.expand_dims(input_eval, 0)

# Empty string to store our results
text_generated = []

# Low temperatures results in more predictable text.
# Higher temperatures results in more surprising text.
# Experiment to find the best setting.
temperature = 1.0

# Evaluation loop.

# Here batch size == 1
model.reset_states()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a multinomial distribution to predict the word returned by the
        # model
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0]

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        predicted_id = sess.run(predicted_id)

        text_generated.append(idx2char[predicted_id])

        print (start_string + ''.join(text_generated))
