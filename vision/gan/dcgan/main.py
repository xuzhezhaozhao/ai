#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import input_data
from generator_net import generator_net
from discriminator_net import discriminator_net

tf.app.flags.DEFINE_string('model_dir', 'model_dir', '')
tf.app.flags.DEFINE_string('export_model_dir', 'export_model_dir', '')
tf.app.flags.DEFINE_string('run_mode', 'train', 'train, eval and sample')
tf.app.flags.DEFINE_string('train_data_path', '', 'train data path')
tf.app.flags.DEFINE_string('eval_data_path', '', 'eval data path')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('max_train_steps', -1, '')
tf.app.flags.DEFINE_integer('epoch', 1, '')

# dataset flags
tf.app.flags.DEFINE_integer('prefetch_size', 1000, '')
tf.app.flags.DEFINE_integer('shuffle_size', 1000, '')
tf.app.flags.DEFINE_bool('shuffle_batch', True, '')
tf.app.flags.DEFINE_integer('map_num_parallel_calls', 1, '')

# log flags
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_integer('save_checkpoints_secs', -1, '')
tf.app.flags.DEFINE_integer('save_checkpoints_steps', -1, '')
tf.app.flags.DEFINE_integer('keep_checkpoint_max', 3, '')
tf.app.flags.DEFINE_integer('log_step_count_steps', 100, '')

tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'learning rate')
tf.app.flags.DEFINE_float('adam_beta1', 0.9, '')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, '')
tf.app.flags.DEFINE_float('opt_epsilon', 1e-8, '')

tf.app.flags.DEFINE_integer('nz', 100, 'latent vector size')
tf.app.flags.DEFINE_integer('ngf', 64, 'initial generator features')
tf.app.flags.DEFINE_integer('ndf', 64, 'initial discriminator features')
tf.app.flags.DEFINE_integer('nc', 3, 'image channels')

opts = tf.app.flags.FLAGS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'


def criterion(label, logits):
    """Cross entropy loss."""
    labels = tf.fill((opts.batch_size, 1), label)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                   logits=logits)
    loss = tf.reduce_mean(loss)
    return loss


def data_iterator():
    input_fn = input_data.build_train_input_fn(opts, opts.train_data_path)
    dataset = input_fn()
    iterator = dataset.make_initializable_iterator()
    return iterator


def build(x):
    real_label = 1.0
    fake_label = 0.0

    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    # train with real
    optimizerD = tf.train.AdamOptimizer(
        learning_rate=opts.learning_rate,
        beta1=opts.adam_beta1,
        beta2=opts.adam_beta2,
        epsilon=opts.opt_epsilon)
    output = discriminator_net(x, True, opts)
    errD_real = criterion(real_label, output)

    # train with fake
    noise = tf.random.normal([opts.batch_size, 1, 1, opts.nz])
    fake = generator_net(noise, True, opts)
    fake_detach = tf.stop_gradient(fake)  # do not update G network
    output = discriminator_net(fake_detach, True, opts)
    errD_fake = criterion(fake_label, output)
    errD = errD_real + errD_fake
    train_op_D = optimizerD.minimize(loss=errD)

    # (2) Update G network: maximize log(D(G(z)))
    optimizerG = tf.train.AdamOptimizer(
        learning_rate=opts.learning_rate,
        beta1=opts.adam_beta1,
        beta2=opts.adam_beta2,
        epsilon=opts.opt_epsilon)
    output = discriminator_net(fake, True, opts)
    errG = criterion(real_label, output)
    train_op_G = optimizerG.minimize(loss=errG)

    return train_op_D, train_op_G, errD_real, errD_fake, errG


def main(_):
    iterator = data_iterator()
    x = iterator.get_next()['data']
    (train_op_D, train_op_G, errD_real, errD_fake, errG) = build(x)

    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(opts.model_dir)
    saver = tf.train.Saver(
        sharded=True,
        max_to_keep=3,
        save_relative_paths=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(iterator.initializer)
        summary_writer.add_graph(sess.graph)
        tf.get_default_graph().finalize()

        for i in xrange(100):
            _, _, e1, e2, e3 = sess.run([train_op_D, train_op_G,
                                         errD_real, errD_fake, errG])
            print("iter {}, errD_real = {:.2f}, errD_fake = {:.2f}, "
                  "errG = {:.2f}".format(i, e1, e2, e3))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
