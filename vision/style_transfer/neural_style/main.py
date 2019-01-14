#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
from neural_style import NeuralStyle

import os
import tensorflow as tf
import numpy as np


tf.app.flags.DEFINE_string('model_dir', 'model_dir', '')
tf.app.flags.DEFINE_string('vgg19_npy_path', '', '')

tf.app.flags.DEFINE_string('style_image_path', '', '')
tf.app.flags.DEFINE_string('content_image_path', '', '')
tf.app.flags.DEFINE_string('output_image_path', '', '')
tf.app.flags.DEFINE_bool('use_init_image', False, '')
tf.app.flags.DEFINE_string('init_image_path', '', '')

# train flags
tf.app.flags.DEFINE_integer('iters', 100, '')
tf.app.flags.DEFINE_float('learning_rate', 10.0, '')
tf.app.flags.DEFINE_float('adam_beta1', 0.9, '')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, '')
tf.app.flags.DEFINE_float('epsilon', 1e-8, '')
tf.app.flags.DEFINE_integer('save_output_steps', 100, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 10, '')
tf.app.flags.DEFINE_integer('save_checkpoints_steps', 100, '')
tf.app.flags.DEFINE_float('content_loss_weight', 0.01, '')
tf.app.flags.DEFINE_float('style_loss_weight', 1.0, '')

opts = tf.app.flags.FLAGS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'


def main(_):
    neural_style = NeuralStyle(opts)
    loss_tensor = neural_style.loss
    content_loss_tensor = neural_style.content_loss
    style_loss_tensor = neural_style.style_loss
    global_step_tensor = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(
        learning_rate=opts.learning_rate,
        beta1=opts.adam_beta1,
        beta2=opts.adam_beta2,
        epsilon=opts.epsilon)
    train_op = optimizer.minimize(loss=loss_tensor,
                                  global_step=global_step_tensor)
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(opts.model_dir)
    saver = tf.train.Saver(
        sharded=True,
        max_to_keep=3,
        save_relative_paths=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer.add_graph(sess.graph)
        tf.get_default_graph().finalize()

        for _ in xrange(opts.iters):
            _, loss, content_loss, style_loss, global_step = sess.run(
                [train_op, loss_tensor,
                 content_loss_tensor, style_loss_tensor,
                 global_step_tensor])
            print("iter {}, loss = {:.2f}, content_loss = {:.2f},"
                  " style_loss = {:.2f}"
                  .format(global_step, loss, content_loss, style_loss))

            if global_step % opts.save_output_steps == 0:
                # save output image
                img = sess.run([neural_style.output_image])
                img = np.squeeze(img, 0)
                output = 'output.' + str(global_step) + '.jpg'
                imsave(output, img)
                print("save output as {}.".format(output))

            if global_step % opts.save_summary_steps == 0:
                summary = sess.run(merged_summary)
                summary_writer.add_summary(summary, global_step)
                summary_writer.flush()

            if global_step % opts.save_checkpoints_steps == 0:
                save_checkpoint(sess, global_step, saver, opts.model_dir)

        # save final output image
        img = sess.run([neural_style.output_image])
        img = np.squeeze(img, 0)
        imsave(opts.output_image_path, img)

        summary_writer.close()


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)


def save_checkpoint(sess, global_step, saver, model_dir):
    ckpt_name = os.path.join(model_dir, 'model.ckpt-{}'.format(global_step))
    ckpt_path = saver.save(sess, ckpt_name)
    print('Model ckpt saved at {}'.format(ckpt_path))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
