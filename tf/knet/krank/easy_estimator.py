#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import copy
import os
from datetime import datetime


class EasyEstimator(object):
    def __init__(self, model_fn, model_dir=None, config=None, params=None):

        if config is None:
            self._config = tf.estimator.RunConfig()
            tf.logging.info('Using default config.')
        else:
            if not isinstance(config, tf.estimator.RunConfig):
                raise ValueError('config must be an instance of RunConfig, '
                                 'but provided %s.' % config)
        self._config = config

        self._model_dir = model_dir or self._config.model_dir
        if self._model_dir is None:
            raise ValueError("model_dir should not be None.")
        if self._config.model_dir is None:
            self._config = self._config.replace(model_dir=self._model_dir)

        tf.logging.info('Using config: %s', str(vars(self._config)))

        self._model_fn = model_fn
        self._params = copy.deepcopy(params or {})

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def params(self):
        return copy.deepcopy(self._params)

    def train(self,
              input_fn,
              hooks=None,
              steps=None,
              max_steps=None):

        loss = self._train_model(input_fn, hooks)
        tf.logging.info('Loss for final step: %s.', loss)
        return self

    def _train_model(self, input_fn, hooks):
        with tf.Graph().as_default() as g:
            tf.set_random_seed(self._config.tf_random_seed)
            global_step_tensor = tf.train.get_or_create_global_step(g)
            features, labels = self._get_features_and_labels_from_input_fn(
                input_fn, tf.estimator.ModeKeys.TRAIN)
            estimator_spec = self._call_model_fn(
                features, labels, tf.estimator.ModeKeys.TRAIN)
            self._merged_summary = tf.summary.merge_all()
            self._writer = tf.summary.FileWriter(self._model_dir)
            self._saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(self._iterator_initializer)
                self._writer.add_graph(sess.graph)

                lastest_path = tf.train.latest_checkpoint(self._model_dir)
                if lastest_path is not None:
                    tf.logging.info('restore model ckpt from {} ...'
                                    .format(lastest_path))
                    self._saver.restore(sess, lastest_path)

                tf.logging.info('{} Start training ...'.format(datetime.now()))

                _, loss = sess.run([estimator_spec.train_op,
                                    estimator_spec.loss])

                global_steps = sess.run(global_step_tensor)
                tf.logging.info('global_steps = {}'.format(global_steps))
                ckpt_name = os.path.join(self._model_dir,
                                        'model.ckpt-{}'.format(global_steps))
                ckpt_path = self._saver.save(sess, ckpt_name)
                tf.logging.info('{} Model ckpt saved at {}'
                                .format(datetime.now(), ckpt_path))

            return loss

    def _get_features_and_labels_from_input_fn(self, input_fn, mode):
        result = self._call_input_fn(input_fn, mode)

        return self._parse_input_fn_result(result)

    def _call_input_fn(self, input_fn, mode):
        with tf.device('/cpu:0'):
            return input_fn()

    def _parse_input_fn_result(self, result):
        iterator = result.make_initializable_iterator()
        self._iterator_initializer = iterator.initializer
        data = iterator.get_next()

        return data[0], data[1]

    def _call_model_fn(self, features, labels, mode):
        tf.logging.info('Calling model_fn.')
        model_fn_results = self._model_fn(
            features=features, labels=labels, params=self.params, mode=mode)
        tf.logging.info('Done calling model_fn.')

        if not isinstance(model_fn_results, tf.estimator.EstimatorSpec):
            raise ValueError('model_fn should return an EstimatorSpec.')

        return model_fn_results
