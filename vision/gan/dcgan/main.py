#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
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
tf.app.flags.DEFINE_integer('save_checkpoints_steps', 100, '')
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

    tf.summary.histogram('inputs', x)

    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    # train with real
    optimizerD = tf.train.AdamOptimizer(
        learning_rate=opts.learning_rate,
        beta1=opts.adam_beta1,
        beta2=opts.adam_beta2,
        epsilon=opts.opt_epsilon)
    output = discriminator_net(x, True, opts)
    errD_real = criterion(real_label, output)
    tf.summary.scalar('errD_real', errD_real)

    # train with fake
    noise = tf.random.normal([opts.batch_size, 1, 1, opts.nz])
    fake = generator_net(noise, True, opts)
    output = discriminator_net(fake, True, opts)
    errD_fake = criterion(fake_label, output)
    errD = errD_real + errD_fake
    tf.summary.scalar('errD_fake', errD_fake)
    tf.summary.scalar('errD', errD)

    # (2) Update G network: maximize log(D(G(z)))
    optimizerG = tf.train.AdamOptimizer(
        learning_rate=opts.learning_rate,
        beta1=opts.adam_beta1,
        beta2=opts.adam_beta2,
        epsilon=opts.opt_epsilon)
    output = discriminator_net(fake, True, opts)
    errG = criterion(real_label, output)
    tf.summary.scalar('errG', errG)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'D' in var.name]
    g_vars = [var for var in t_vars if 'G' in var.name]
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op_D = optimizerD.minimize(loss=errD, var_list=d_vars)
        train_op_G = optimizerG.minimize(loss=errG, var_list=g_vars)

    return train_op_D, train_op_G, errD_real, errD_fake, errG


def main(_):
    iterator = data_iterator()
    x = iterator.get_next()['data']
    (train_op_D, train_op_G, errD_real, errD_fake, errG) = build(x)

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
        for step in xrange(1000):
            _, _, e1, e2, e3 = sess.run([train_op_D, train_op_G,
                                         errD_real, errD_fake, errG])

            if step % opts.log_step_count_steps == 0:
                print("step {}, errD_real = {:.5f}, errD_fake = {:.5f}, "
                      "errG = {:.5f}, elapsed {:.2f} s"
                      .format(step, e1, e2, e3, time.time() - start))
                start = time.time()

            if step % opts.save_summary_steps == 0:
                summary = sess.run(merged_summary)
                summary_writer.add_summary(summary, step)
                summary_writer.flush()

            if step % opts.save_checkpoints_steps == 0:
                save_checkpoint(sess, step, saver, opts.model_dir)
        summary_writer.close()


def save_checkpoint(sess, global_step, saver, model_dir):
    ckpt_name = os.path.join(model_dir, 'model.ckpt-{}'.format(global_step))
    ckpt_path = saver.save(sess, ckpt_name)
    print('Model ckpt saved at {}'.format(ckpt_path))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
