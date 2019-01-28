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
tf.app.flags.DEFINE_float('adam_beta1', 0.9, '')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, '')
tf.app.flags.DEFINE_float('gp_lambda', 10.0, '')
tf.app.flags.DEFINE_integer('Diters', 5, '')
tf.app.flags.DEFINE_integer('img_size', 64, '')
tf.app.flags.DEFINE_integer('nz', 100, 'latent vector size')
tf.app.flags.DEFINE_integer('ngf', 64, 'initial generator features')
tf.app.flags.DEFINE_integer('ndf', 64, 'initial discriminator features')
tf.app.flags.DEFINE_integer('nc', 3, 'image channels')

tf.app.flags.DEFINE_integer('num_categorical', 10, '')
tf.app.flags.DEFINE_integer('num_continuous', 2, '')
tf.app.flags.DEFINE_float('categorical_weight', 1.0, '')
tf.app.flags.DEFINE_float('continuous_weight', 1.0, '')


opts = tf.app.flags.FLAGS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'


def data_iterator():
    input_fn = input_data.build_train_input_fn(opts, opts.train_data_path)
    dataset = input_fn()
    iterator = dataset.make_initializable_iterator()
    return iterator


def build(x):
    # generate imcompression noise and latent code
    inoise = tf.random.uniform([opts.batch_size, opts.nz], -1.0, 1.0)

    categorical_code = tf.random_uniform((opts.batch_size, 1),
                                         0, opts.num_categorical,
                                         dtype=tf.int32)
    categorical_code = tf.one_hot(categorical_code, opts.num_categorical)
    categorical_code = tf.reshape(categorical_code, (opts.batch_size, -1))
    continuous_code = tf.random.normal((opts.batch_size, opts.num_continuous))

    noise = tf.concat([inoise, categorical_code, continuous_code], 1)
    noise = tf.reshape(noise, (opts.batch_size, 1, 1, -1))
    tf.logging.info('noise: {}'.format(noise))

    # (1) Update D network
    # train with real
    disc_real, _ = discriminator_net(x, True, opts)
    errD_real = -tf.reduce_mean(disc_real)

    # train with fake
    fake = generator_net(noise, True, opts)
    disc_fake, q = discriminator_net(fake, True, opts)
    errD_fake = tf.reduce_mean(disc_fake)

    # gradients penaty
    alpha = tf.random_uniform(shape=[opts.batch_size, 1, 1, 1])
    interpolates = alpha * x + (1 - alpha) * fake
    disc_inter, _ = discriminator_net(interpolates, True, opts)
    gradients = tf.gradients(disc_inter, [interpolates])[0]
    gradients = tf.reshape(gradients, [opts.batch_size, -1])
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
    gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
    errD = errD_real + errD_fake + opts.gp_lambda * gradient_penalty
    optimizerD = tf.train.AdamOptimizer(
        learning_rate=opts.learning_rate,
        beta1=opts.adam_beta1,
        beta2=opts.adam_beta2)

    # (2) Update G network
    errG = -errD_fake
    optimizerG = tf.train.AdamOptimizer(
        learning_rate=opts.learning_rate,
        beta1=opts.adam_beta1,
        beta2=opts.adam_beta2)

    # (3) Update Q network
    # categorical loss
    err_categorical = tf.losses.softmax_cross_entropy(
        categorical_code, q[:, :opts.num_categorical])

    # continuous loss
    err_continuous = tf.losses.mean_squared_error(
        labels=continuous_code, predictions=q[:, opts.num_categorical:])

    errQ = opts.categorical_weight * err_categorical + \
        opts.continuous_weight * err_continuous
    optimizerQ = tf.train.AdamOptimizer(
        learning_rate=opts.learning_rate,
        beta1=opts.adam_beta1,
        beta2=opts.adam_beta2)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'Discriminator' in var.name]
    g_vars = [var for var in t_vars if 'Generator' in var.name]
    q_vars = [var for var in t_vars if 'QNet' in var.name]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op_D = optimizerD.minimize(errD, var_list=d_vars)
        train_op_G = optimizerG.minimize(errG, var_list=g_vars)
        train_op_Q = optimizerQ.minimize(errQ, var_list=q_vars + g_vars)

    # add summary
    tf.summary.scalar('errD', errD)
    tf.summary.scalar('errG', errG)
    tf.summary.scalar('errQ', errQ)
    tf.summary.scalar('errD_fake', errD_fake)
    tf.summary.scalar('errD_real', errD_real)
    tf.summary.scalar('gradient_penalty', gradient_penalty)
    tf.summary.scalar('err_categorical', err_categorical)
    tf.summary.scalar('err_continuous', err_continuous)

    add_summary_img('input', x)
    add_summary_img('fake', fake)
    tf.summary.histogram('input_h', x)
    tf.summary.histogram('fake_h', fake)

    return train_op_D, train_op_G, train_op_Q, errD, errG, errQ


def add_summary_img(name, img):
    tf.summary.image(name, tf.cast(tf.clip_by_value(
        input_data.invert_norm(img), 0, 255), tf.uint8))


def train():
    iterator = data_iterator()
    x = iterator.get_next()['data']
    (train_op_D, train_op_G, train_op_Q, errD, errG, errQ) = build(x)

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
                e1, e2, e3 = sess.run([errD, errG, errQ])
                tf.logging.info(
                    "step {}, errD = {:.5f}, errG = {:.5f}, errQ = {:.5f}, "
                    "elapsed {:.2f} s"
                    .format(step, e1, e2, e3, time.time() - start))
                start = time.time()

            Diters = opts.Diters
            if step < 25 or step % 500 == 0:
                Diters = 100
            for _ in range(Diters):
                sess.run(train_op_D)
            sess.run(train_op_G)
            sess.run(train_op_Q)

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
        batch_size = 12
        inoise = tf.random.uniform([batch_size, opts.nz], -1.0, 1.0)
        categorical_code = tf.fill((batch_size, 1), 1)
        categorical_code = tf.one_hot(categorical_code, opts.num_categorical)
        categorical_code = tf.reshape(categorical_code, (batch_size, -1))
        continuous_code = tf.random.normal((batch_size, opts.num_continuous))
        noise = tf.concat([inoise, categorical_code, continuous_code], 1)
        noise = tf.reshape(noise, (batch_size, 1, 1, -1))
        print(noise)
        # set training True for good quality image
        fake = generator_net(noise, True, opts)
        checkpoint_path = opts.sample_checkpoint_path
        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
            output = sess.run(fake)
            output = input_data.invert_norm(output)
            for idx, img in enumerate(output):
                if opts.nc == 1:
                    img = np.squeeze(img, -1)
                utils.imsave(str(idx + 1) + '_' + filename, img)


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
