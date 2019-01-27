#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np

from generator_net import generator_net
from discriminator_net import discriminator_net
import utils
import input_data

tf.app.flags.DEFINE_string('model_dir', 'model_dir', '')
tf.app.flags.DEFINE_string('export_model_dir', 'export_model_dir', '')
tf.app.flags.DEFINE_string('sample_checkpoint_path', '', '')
tf.app.flags.DEFINE_string('run_mode', 'train', 'train, eval and sample')
tf.app.flags.DEFINE_string('train_data_path', '', 'train data path')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('max_train_steps', 1000, '')

# dataset flags
tf.app.flags.DEFINE_integer('prefetch_size', 1000, '')
tf.app.flags.DEFINE_integer('shuffle_size', 1000, '')
tf.app.flags.DEFINE_bool('shuffle_batch', True, '')
tf.app.flags.DEFINE_integer('map_num_parallel_calls', 1, '')

# log flags
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_integer('save_checkpoints_steps', 100, '')
tf.app.flags.DEFINE_integer('keep_checkpoint_max', 3, '')
tf.app.flags.DEFINE_integer('log_step_count_steps', 100, '')
tf.app.flags.DEFINE_integer('save_output_steps', 500, '')

tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'learning rate')
tf.app.flags.DEFINE_integer('Diters', 1, '')
tf.app.flags.DEFINE_integer('img_size', 64, '')
tf.app.flags.DEFINE_integer('nz', 100, 'latent vector size')
tf.app.flags.DEFINE_integer('ngf', 64, 'initial generator features')
tf.app.flags.DEFINE_integer('ndf', 64, 'initial discriminator features')
tf.app.flags.DEFINE_integer('nc', 3, 'image channels')

opts = tf.app.flags.FLAGS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'

fixed_noise = np.random.uniform(-1.0, 1.0,
                                size=[1, 1, 1, opts.nz]).astype(np.float32)


def data_iterator():
    input_fn = input_data.build_train_input_fn(opts, opts.train_data_path)
    dataset = input_fn()
    iterator = dataset.make_initializable_iterator()
    return iterator


def build(x):
    # (1) Update D network
    # train with real
    disc_real = discriminator_net(x, True, opts)
    errD_real = tf.reduce_mean((disc_real - 1.)**2)

    # train with fake
    noise = tf.random.uniform([opts.batch_size, 1, 1, opts.nz], -1.0, 1.0)
    fake = generator_net(noise, True, opts)
    disc_fake = discriminator_net(fake, True, opts)
    errD_fake = tf.reduce_mean((disc_fake - 0.)**2)
    errD = errD_real + errD_fake
    optimizerD = tf.train.RMSPropOptimizer(learning_rate=opts.learning_rate)

    # (2) Update G network
    errG = tf.reduce_mean((disc_fake - 1.)**2)
    optimizerG = tf.train.RMSPropOptimizer(learning_rate=opts.learning_rate)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'Discriminator' in var.name]
    g_vars = [var for var in t_vars if 'Generator' in var.name]
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op_D = optimizerD.minimize(errD, var_list=d_vars)
        train_op_G = optimizerG.minimize(errG, var_list=g_vars)

    # add summary
    tf.summary.scalar('errD', errD)
    tf.summary.scalar('errG', errG)
    tf.summary.scalar('errD_fake', errD_fake)
    tf.summary.scalar('errD_real', errD_real)

    add_summary_img('input', x)
    add_summary_img('fake', fake)
    tf.summary.histogram('input_h', x)
    tf.summary.histogram('fake_h', fake)

    return train_op_D, train_op_G, errD, errG, errD_real, errD_fake


def add_summary_img(name, img):
    tf.summary.image(name, tf.cast(tf.clip_by_value(
        input_data.invert_norm(img), 0, 255), tf.uint8))


def train():
    iterator = data_iterator()
    x = iterator.get_next()['data']
    (train_op_D, train_op_G, errD, errG, errD_real, errD_fake) = build(x)

    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(opts.model_dir)
    saver = tf.train.Saver(
        sharded=True,
        max_to_keep=opts.keep_checkpoint_max,
        save_relative_paths=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(iterator.initializer)
        summary_writer.add_graph(sess.graph)
        tf.get_default_graph().finalize()

        start = time.time()
        for step in xrange(opts.max_train_steps):
            if step % opts.log_step_count_steps == 0:
                e1, e2, e3, e4 = sess.run([errD, errG, errD_real, errD_fake])
                tf.logging.info(
                    "step {}, errD = {:.5f}, errG = {:.5f}, errD_real = {:.5f}"
                    ", errD_fake = {:.5f}, elapsed {:.2f} s"
                    .format(step, e1, e2, e3, e4, time.time() - start))
                start = time.time()

            for _ in range(opts.Diters):
                sess.run(train_op_D)
            sess.run(train_op_G)

            if step % opts.save_summary_steps == 0:
                summary = sess.run(merged_summary)
                summary_writer.add_summary(summary, step)
                summary_writer.flush()

            if step % opts.save_checkpoints_steps == 0:
                save_checkpoint(sess, step, saver, opts.model_dir)

            if step % opts.save_output_steps == 0:
                filename = 'output.{}.jpg'.format(step)
                sample(filename)

        summary_writer.close()


def sample(filename='output.jpg'):
    with tf.Graph().as_default():
        # set training True for good quality image
        fake = generator_net(tf.constant(fixed_noise), True, opts)
        checkpoint_path = opts.sample_checkpoint_path
        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
            output = sess.run(fake)
            output = input_data.invert_norm(output)
            output = np.squeeze(output, 0)
            if opts.nc == 1:
                output = np.squeeze(output, -1)
            utils.imsave(filename, output)


def main(_):
    if opts.run_mode == 'train':
        train()
    elif opts.run_mode == 'sample':
        sample()


def save_checkpoint(sess, global_step, saver, model_dir):
    ckpt_name = os.path.join(model_dir, 'model.ckpt-{}'.format(global_step))
    ckpt_path = saver.save(sess, ckpt_name)
    tf.logging.info('Model ckpt saved at {}'.format(ckpt_path))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
