#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import copy
import os
import six

from google.protobuf import message
from datetime import datetime
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.python.training import device_setter
from tensorflow.core.framework import summary_pb2
from tensorflow.python.eager import context
from tensorflow.python.estimator.export import export as export_helpers
from tensorflow.python.estimator.export import export_output
from tensorflow.python.estimator import model_fn as model_fn_lib


class EasyEstimator(object):
    def __init__(self, model_fn, model_dir=None, config=None, params=None,
                 warm_start_from=None):

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

        if self._config.session_config is None:
            rewrite_opts = RewriterConfig(
                meta_optimizer_iterations=RewriterConfig.ONE)
            graph_opts = config_pb2.GraphOptions(rewrite_options=rewrite_opts)
            self._session_config = config_pb2.ConfigProto(
                allow_soft_placement=True, graph_options=graph_opts)
        else:
            self._session_config = self._config.session_config

        self._device_fn = (
            self._config.device_fn or _get_replica_device_setter(self._config))

        self._model_fn = model_fn
        self._params = copy.deepcopy(params or {})
        self._warm_start_settings = _get_default_warm_start_settings(
            warm_start_from)

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def params(self):
        return copy.deepcopy(self._params)

    @property
    def config(self):
        return copy.deepcopy(self._config)

    def train(self,
              input_fn,
              steps=None,
              max_steps=None):
        with context.graph_mode():
            if (steps is not None) and (max_steps is not None):
                raise ValueError(
                    'Can not provide both steps and max_steps.')
            if steps is not None and steps <= 0:
                raise ValueError(
                    'Must specify steps > 0, given: {}'.format(steps))
            if max_steps is not None and max_steps <= 0:
                raise ValueError('Must specify max_steps > 0, given: {}'
                                 .format(max_steps))

            if max_steps is not None:
                start_step = _load_global_step_from_checkpoint_dir(
                    self._model_dir)
                if max_steps <= start_step:
                    tf.logging.info('Skipping training since max_steps '
                                    'has already saved.')
                    return self

            loss = self._train_model(input_fn)
            tf.logging.info('Loss for final step: %s.', loss)
            return self

    def evaluate(self,
                 input_fn,
                 steps=None,
                 hooks=None,
                 checkpoint_path=None,
                 name=None):
        with context.graph_mode():
            hooks = _check_hooks_type(hooks)
            hooks.extend(self._convert_eval_steps_to_hooks(steps))

            if not checkpoint_path:
                latest_path = tf.train.latest_checkpoint(self._model_dir)
                if not latest_path:
                    raise ValueError('Could not find trained model in'
                                     'model_dir: {}'.format(self._model_dir))
                checkpoint_path = latest_path

            with tf.Graph().as_default():
                (scaffold, update_op,
                 eval_dict, all_hooks) = self._evaluate_build_graph(
                     input_fn, hooks, checkpoint_path)

                return self._evaluate_run(
                    checkpoint_path=checkpoint_path,
                    scaffold=scaffold,
                    update_op=update_op,
                    eval_dict=eval_dict,
                    all_hooks=all_hooks,
                    output_dir=self.eval_dir(name))

    def eval_dir(self, name):
        return os.path.join(self._model_dir,
                            'eval' if not name else 'eval_' + name)

    def predict(self,
                input_fn,
                predict_keys=None,
                hooks=None,
                checkpoint_path=None,
                yield_single_examples=True):
        with context.graph_mode():
            hooks = _check_hooks_type(hooks)
            # Check that model has been trained.
            if not checkpoint_path:
                checkpoint_path = tf.train.latest_checkpoint(self._model_dir)
            if not checkpoint_path:
                tf.logging.info(
                    'Could not find trained model in model_dir: {},'
                    ' running initialization to predict.'
                    .format(self._model_dir))
            with tf.Graph().as_default() as g:
                tf.set_random_seed(self._config.tf_random_seed)
                tf.train.create_global_step(graph=g)
                features, input_hooks = self._get_features_from_input_fn(
                    input_fn, tf.estimator.ModeKeys.PREDICT)

                estimator_spec = self._call_model_fn(
                    features, None, tf.estimator.ModeKeys.PREDICT, self.config)

                # Call to warm_start has to be after model_fn is called.
                self._maybe_warm_start(checkpoint_path)

                predictions = self._extract_keys(
                    estimator_spec.predictions, predict_keys)
                all_hooks = list(input_hooks)
                all_hooks.extend(hooks)
                all_hooks.extend(list(estimator_spec.prediction_hooks or []))
                with tf.train.MonitoredSession(
                        session_creator=tf.train.ChiefSessionCreator(
                            checkpoint_filename_with_path=checkpoint_path,
                            master=self._config.master,
                            scaffold=estimator_spec.scaffold,
                            config=self._session_config),
                        hooks=all_hooks) as mon_sess:
                    while not mon_sess.should_stop():
                        preds_evaluated = mon_sess.run(predictions)
                        if not yield_single_examples:
                            yield preds_evaluated
                        elif not isinstance(predictions, dict):
                            for pred in preds_evaluated:
                                yield pred
                        else:
                            for i in range(self._extract_batch_length(
                                    preds_evaluated)):
                                yield {
                                    key: value[i]
                                    for key, value in six.iteritems(
                                        preds_evaluated)
                                }

    def export_savedmodel(
            self, export_dir_base, serving_input_receiver_fn,
            assets_extra=None,
            as_text=False,
            checkpoint_path=None,
            strip_default_attrs=False):
        return self._export_saved_model_for_mode(
            export_dir_base,
            serving_input_receiver_fn,
            assets_extra=assets_extra,
            as_text=as_text,
            checkpoint_path=checkpoint_path,
            strip_default_attrs=strip_default_attrs,
            mode=tf.estimator.ModeKeys.PREDICT)

    def _export_saved_model_for_mode(
            self, export_dir_base, input_receiver_fn,
            assets_extra=None,
            as_text=False,
            checkpoint_path=None,
            strip_default_attrs=False,
            mode=tf.estimator.ModeKeys.PREDICT):
        if not input_receiver_fn:
            raise ValueError('An input_receiver_fn must be defined.')

        input_receiver_fn_map = {mode: input_receiver_fn}

        return self._export_all_saved_models(
            export_dir_base,
            input_receiver_fn_map,
            assets_extra=assets_extra,
            as_text=as_text,
            checkpoint_path=checkpoint_path,
            strip_default_attrs=strip_default_attrs)

    def _export_all_saved_models(
            self, export_dir_base, input_receiver_fn_map,
            assets_extra=None,
            as_text=False,
            checkpoint_path=None,
            strip_default_attrs=False):

        with context.graph_mode():
            if not checkpoint_path:
                # Locate the latest checkpoint
                checkpoint_path = tf.train.latest_checkpoint(self._model_dir)
            if not checkpoint_path:
                raise ValueError(
                    "Couldn't find trained model at %s." % self._model_dir)

            export_dir = export_helpers.get_timestamped_export_dir(
                export_dir_base)
            temp_export_dir = export_helpers.get_temp_export_dir(export_dir)
            builder = tf.saved_model.builder.SavedModelBuilder(temp_export_dir)

            save_variables = True

            if input_receiver_fn_map.get(tf.estimator.ModeKeys.TRAIN):
                self._add_meta_graph_for_mode(
                    builder, input_receiver_fn_map, checkpoint_path,
                    strip_default_attrs, save_variables,
                    mode=tf.estimator.ModeKeys.TRAIN)
                save_variables = False
            if input_receiver_fn_map.get(tf.estimator.ModeKeys.EVAL):
                self._add_meta_graph_for_mode(
                    builder, input_receiver_fn_map, checkpoint_path,
                    strip_default_attrs, save_variables,
                    mode=tf.estimator.ModeKeys.EVAL)
                save_variables = False
            if input_receiver_fn_map.get(tf.estimator.ModeKeys.PREDICT):
                self._add_meta_graph_for_mode(
                    builder, input_receiver_fn_map, checkpoint_path,
                    strip_default_attrs, save_variables,
                    mode=tf.estimator.ModeKeys.PREDICT)
            save_variables = False

            if save_variables:
                raise ValueError('No valid modes for exporting found. Got {}.'
                                 .format(input_receiver_fn_map.keys()))

            builder.save(as_text)

            # Add the extra assets
            if assets_extra:
                assets_extra_path = os.path.join(
                    tf.compat.as_bytes(temp_export_dir),
                    tf.compat.as_bytes('assets.extra'))
                for dest_relative, source in assets_extra.items():
                    dest_absolute = os.path.join(
                        tf.compat.as_bytes(assets_extra_path),
                        tf.compat.as_bytes(dest_relative))
                    dest_path = os.path.dirname(dest_absolute)
                    tf.gfile.MakeDirs(dest_path)
                    tf.gfile.Copy(source, dest_absolute)

            tf.gfile.Rename(temp_export_dir, export_dir)
            return export_dir

    def _add_meta_graph_for_mode(self,
                                 builder,
                                 input_receiver_fn_map,
                                 checkpoint_path,
                                 strip_default_attrs,
                                 save_variables=True,
                                 mode=tf.estimator.ModeKeys.PREDICT,
                                 export_tags=None,
                                 check_variables=True):
        if export_tags is None:
            export_tags = model_fn_lib.EXPORT_TAG_MAP[mode]
        input_receiver_fn = input_receiver_fn_map[mode]

        with tf.Graph().as_default() as g:
            tf.train.create_global_step(g)
            tf.set_random_seed(self._config.tf_random_seed)

            input_receiver = input_receiver_fn()

            # Call the model_fn and collect the export_outputs.
            estimator_spec = self._call_model_fn(
                features=input_receiver.features,
                labels=getattr(input_receiver, 'labels', None),
                mode=mode)

            export_outputs = self._get_export_outputs_for_spec(estimator_spec)

            # Build the SignatureDefs from receivers and all outputs
            signature_def_map = export_helpers.build_all_signature_defs(
                input_receiver.receiver_tensors,
                export_outputs,
                getattr(input_receiver, 'receiver_tensors_alternatives', None),
                serving_only=(mode == model_fn_lib.ModeKeys.PREDICT))

            with tf.Session(config=self._session_config) as session:

                local_init_op = (
                    estimator_spec.scaffold.local_init_op or
                    tf.train.Scaffold.default_local_init_op())

                graph_saver = estimator_spec.scaffold.saver or tf.train.Saver(
                    sharded=True)

                if save_variables and not check_variables:
                    raise ValueError('If `save_variables` is `True, '
                                     '`check_variables` must not be `False`.')
                if check_variables:
                    try:
                        graph_saver.restore(session, checkpoint_path)
                    except tf.errors.NotFoundError as e:
                        msg = ('Could not load all requested variables '
                               'from checkpoint. Please make sure your '
                               'model_fn does not expect variables '
                               'that were not saved in the checkpoint.\n\n'
                               'Encountered error with mode `{}` while '
                               'restoring checkpoint from: `{}`. '
                               'Full Traceback:\n\n{}').format(
                            mode, checkpoint_path, e)
                        raise ValueError(msg)

                builder._add_train_op(estimator_spec.train_op)

                meta_graph_kwargs = dict(
                    tags=export_tags,
                    signature_def_map=signature_def_map,
                    assets_collection=tf.get_collection(
                        tf.GraphKeys.ASSET_FILEPATHS),
                    strip_default_attrs=strip_default_attrs,
                    legacy_init_op=local_init_op,
                    saver=graph_saver)

                if save_variables:
                    builder.add_meta_graph_and_variables(
                        session, **meta_graph_kwargs)
                else:
                    builder.add_meta_graph(**meta_graph_kwargs)

    def _get_export_outputs_for_spec(self, estimator_spec):
        mode = estimator_spec.mode
        if mode == model_fn_lib.ModeKeys.PREDICT:
            outputs = estimator_spec.export_outputs
        else:
            if mode == tf.estimator.ModeKeys.TRAIN:
                output_class = export_output.TrainOutput
            elif mode == tf.estimator.ModeKeys.EVAL:
                output_class = export_output.EvalOutput
            else:
                raise ValueError(
                    'Export output type not found for mode: {}'.format(mode))

            export_out = output_class(
                loss=estimator_spec.loss,
                predictions=estimator_spec.predictions,
                metrics=estimator_spec.eval_metric_ops)
            outputs = {mode: export_out}

        return outputs

    def _train_model(self, input_fn):
        with tf.Graph().as_default() as g:
            tf.set_random_seed(self._config.tf_random_seed)
            global_step_tensor = tf.train.get_or_create_global_step(g)
            features, labels, _ = self._get_features_and_labels_from_input_fn(
                input_fn, tf.estimator.ModeKeys.TRAIN)
            estimator_spec = self._call_model_fn(
                features, labels, tf.estimator.ModeKeys.TRAIN)
            merged_summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self._model_dir)
            saver = tf.train.Saver(
                sharded=True,
                max_to_keep=self._config.keep_checkpoint_max,
                save_relative_paths=True)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(self._iterator_initializer)
                summary_writer.add_graph(sess.graph)

                self._maybe_restore_model(sess, saver)

                tf.logging.info('{} Start training ...'.format(datetime.now()))

                while True:
                    try:
                        _, loss = sess.run([estimator_spec.train_op,
                                            estimator_spec.loss])
                        global_steps = sess.run(global_step_tensor)
                        break
                    except tf.errors.OutOfRangeError:
                        break

                    if self._check_global_steps(
                            global_steps, self._config.log_step_count_steps):
                        self._logging(sess, global_steps, loss)

                    if self._check_global_steps(
                            global_steps, self._config.save_checkpoints_steps):
                        self._save_ckpt(sess, global_steps, saver)

                    if self._check_global_steps(
                            global_steps, self._config.save_summary_steps):
                        self._save_summary(
                            sess, global_steps, summary_writer, merged_summary)

                self._logging(sess, global_steps, loss)
                self._save_ckpt(sess, global_steps, saver)
                summary_writer.close()

            return loss

    def _get_features_and_labels_from_input_fn(self, input_fn, mode):
        result = self._call_input_fn(input_fn, mode)

        return self._parse_input_fn_result(result)

    def _get_features_from_input_fn(self, input_fn, mode):
        """Extracts the `features` from return values of `input_fn`."""
        result = self._call_input_fn(input_fn, mode)
        result, _, hooks = self._parse_input_fn_result(result)
        return result, hooks

    def _call_input_fn(self, input_fn, mode):
        with tf.device('/cpu:0'):
            return input_fn()

    def _parse_input_fn_result(self, result):
        iterator = result.make_initializable_iterator()
        self._iterator_initializer = iterator.initializer
        data = iterator.get_next()
        input_hooks = [_DatasetInitializerHook(iterator)]

        if isinstance(data, (list, tuple)):
            if len(data) != 2:
                raise ValueError('input_fn should return (features, labels)'
                                 ' as a len 2 tuple.')
            return data[0], data[1], input_hooks
        return data, None, input_hooks

        return data[0], data[1], input_hooks

    def _call_model_fn(self, features, labels, mode):
        tf.logging.info('Calling model_fn.')
        model_fn_results = self._model_fn(
            features=features, labels=labels, params=self.params, mode=mode)
        tf.logging.info('Done calling model_fn.')

        if not isinstance(model_fn_results, tf.estimator.EstimatorSpec):
            raise ValueError('model_fn should return an EstimatorSpec.')

        return model_fn_results

    def _maybe_restore_model(self, sess, saver):
        lastest_path = tf.train.latest_checkpoint(self._model_dir)
        if lastest_path is not None:
            tf.logging.info('restore model ckpt from {} ...'
                            .format(lastest_path))
            saver.restore(sess, lastest_path)

    def _logging(self, sess, global_steps, loss):
        tf.logging.info('{} global_steps = {}, loss = {}'
                        .format(datetime.now(), global_steps, loss))

    def _save_ckpt(self, sess, global_steps, saver):
        ckpt_name = os.path.join(self._model_dir,
                                 'model.ckpt-{}'.format(global_steps))
        ckpt_path = saver.save(sess, ckpt_name)
        tf.logging.info('{} Model ckpt saved at {}'
                        .format(datetime.now(), ckpt_path))

    def _save_summary(self, sess, global_steps, writer, merged_summary):
        merged_summary = sess.run(merged_summary)
        writer.add_summary(merged_summary, global_steps)
        writer.flush()
        tf.logging.info('{} Summary saved.'.format(datetime.now()))

    def _check_global_steps(self, global_steps, steps):
        return True if global_steps % steps == 0 else False

    def _evaluate_build_graph(self, input_fn, hooks=None,
                              checkpoint_path=None):
        """Builds the graph and related hooks to run evaluation."""
        tf.set_random_seed(self._config.tf_random_seed)
        global_step_tensor = tf.train.get_or_create_global_step(
            tf.get_default_graph())
        (features, labels,
         input_hooks) = self._get_features_and_labels_from_input_fn(
             input_fn, tf.estimator.ModeKeys.EVAL)
        estimator_spec = self._call_model_fn(
            features, labels, tf.estimator.ModeKeys.EVAL)

        # Call to warm_start has to be after model_fn is called.
        self._maybe_warm_start(checkpoint_path)

        if 'loss' in estimator_spec.eval_metric_ops:
            raise ValueError(
                'Metric with name "loss" is not allowed, because Estimator '
                'already defines a default metric with the same name.')

        estimator_spec.eval_metric_ops[
            'loss'] = tf.metrics.mean(estimator_spec.loss)

        update_op, eval_dict = _extract_metric_update_ops(
            estimator_spec.eval_metric_ops)

        if tf.GraphKeys.GLOBAL_STEP in eval_dict:
            raise ValueError(
                'Metric with name `global_step` is not allowed, because'
                'Estimator already defines a default metric with the same'
                'name.')
        eval_dict[tf.GraphKeys.GLOBAL_STEP] = global_step_tensor

        all_hooks = list(input_hooks)
        all_hooks.extend(hooks)
        all_hooks.extend(list(estimator_spec.evaluation_hooks or []))

        return estimator_spec.scaffold, update_op, eval_dict, all_hooks

    def _evaluate_run(self, checkpoint_path, scaffold, update_op,
                      eval_dict, all_hooks, output_dir):
        eval_results = tf.contrib.training.evaluate_once(
            checkpoint_path=checkpoint_path,
            master=self._config.evaluation_master,
            scaffold=scaffold,
            eval_ops=update_op,
            final_ops=eval_dict,
            hooks=all_hooks,
            config=self._session_config)

        current_global_step = eval_results[tf.GraphKeys.GLOBAL_STEP]

        _write_dict_to_summary(
            output_dir=output_dir,
            dictionary=eval_results,
            current_global_step=current_global_step)

        if checkpoint_path:
            _write_checkpoint_path_to_summary(
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                current_global_step=current_global_step)

        return eval_results

    def _convert_eval_steps_to_hooks(self, steps):
        if steps is None:
            return []

        if steps <= 0:
            raise ValueError('Must specify steps > 0, given: {}'.format(steps))
        return [tf.contrib.training.StopAfterNEvalsHook(num_evals=steps)]

    def _maybe_warm_start(self, checkpoint_path):
        if not checkpoint_path and self._warm_start_settings:
            tf.logging.info('Warm-starting with WarmStartSettings: %s' %
                            (self._warm_start_settings,))
            tf.train.warm_start(*self._warm_start_settings)

    def _extract_keys(self, predictions, predict_keys):
        """Extracts `predict_keys` from `predictions`."""
        if not predict_keys:
            return predictions
        if not isinstance(predictions, dict):
            raise ValueError('predict_keys argument is not valid in case of'
                             ' non-dict predictions.')
        existing_keys = predictions.keys()
        predictions = {
            key: value
            for key, value in six.iteritems(predictions) if key in predict_keys
        }
        if not predictions:
            raise ValueError('Expected to run at least one output from %s, '
                             'provided %s.' % (existing_keys, predict_keys))
        return predictions

    def _extract_batch_length(self, preds_evaluated):
        """Extracts batch length of predictions."""
        batch_length = None
        for key, value in six.iteritems(preds_evaluated):
            batch_length = batch_length or value.shape[0]
            if value.shape[0] != batch_length:
                raise ValueError('Batch length of predictions should be same.'
                                 ' %s has different batch length than others.'
                                 % key)
        return batch_length


def _extract_metric_update_ops(eval_dict):
    """Separate update operations from metric value operations."""
    update_ops = []
    value_ops = {}
    # Sort metrics lexicographically so graph is identical every time.
    for name, metric_ops in sorted(six.iteritems(eval_dict)):
        value_ops[name] = metric_ops[0]
        update_ops.append(metric_ops[1])

    if update_ops:
        update_op = tf.group(*update_ops)
    else:
        update_op = None

    return update_op, value_ops


def _check_hooks_type(hooks):
    """Returns hooks if all are SessionRunHook, raises TypeError otherwise."""
    hooks = list(hooks or [])
    for h in hooks:
        if not isinstance(h, tf.train.SessionRunHook):
            raise TypeError(
                'Hooks must be a SessionRunHook, given: {}'.format(h))
    return hooks


def _get_default_warm_start_settings(warm_start_from):

    if warm_start_from is None:
        return None
    if isinstance(warm_start_from, (six.string_types, six.binary_type)):
        # Infer that this is a SavedModel if export_path +
        # 'variables/variables.index' exists, and if so, construct the
        # WarmStartSettings pointing to export_path + 'variables/variables'.
        if tf.gfile.Exists(os.path.join(tf.compat.as_bytes(warm_start_from),
                                        tf.compat.as_bytes(
                                            'variables/variables.index'))):
            tf.logging.info('Warm-starting from a SavedModel')
            return tf.estimator.WarmStartSettings(
                ckpt_to_initialize_from=os.path.join(
                    tf.compat.as_bytes(warm_start_from),
                    tf.compat.as_bytes('{}/{}'.format(
                        tf.saved_model.constants.VARIABLES_DIRECTORY,
                        tf.saved_model.constants.VARIABLES_FILENAME))))
        return tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=warm_start_from)
    elif isinstance(warm_start_from, tf.estimator.WarmStartSettings):
        return warm_start_from
    else:
        raise ValueError('warm_start_from must be a string or a '
                         'WarmStartSettings, instead got {}'
                         .format(type(warm_start_from)))


class _DatasetInitializerHook(tf.train.SessionRunHook):
    """Creates a SessionRunHook that initializes the passed iterator."""

    def __init__(self, iterator):
        self._iterator = iterator

    def begin(self):
        self._initializer = self._iterator.initializer

    def after_create_session(self, session, coord):
        del coord
        session.run(self._initializer)


def _get_replica_device_setter(config):
    if config.task_type:
        worker_device = '/job:%s/task:%d' % (config.task_type, config.task_id)
    else:
        worker_device = '/job:worker'

    if config.num_ps_replicas > 0:
        return tf.train.replica_device_setter(
            ps_tasks=config.num_ps_replicas,
            worker_device=worker_device,
            merge_devices=True,
            ps_ops=list(device_setter.STANDARD_PS_OPS),
            cluster=config.cluster_spec)
    else:
        return None


def _dict_to_str(dictionary):
    """Get a `str` representation of a `dict`.

    Args:
      dictionary: The `dict` to be represented as `str`.

    Returns:
      A `str` representing the `dictionary`.
    """
    return ', '.join('%s = %s' % (k, v)
                     for k, v in sorted(six.iteritems(dictionary))
                     if not isinstance(v, six.binary_type))


def _write_dict_to_summary(output_dir,
                           dictionary,
                           current_global_step):
    """Writes a `dict` into summary file in given output directory.

    Args:
      output_dir: `str`, directory to write the summary file in.
      dictionary: the `dict` to be written to summary file.
      current_global_step: `int`, the current global step.
    """
    tf.logging.info('Saving dict for global step %d: %s', current_global_step,
                    _dict_to_str(dictionary))
    summary_writer = tf.summary.FileWriterCache.get(output_dir)
    summary_proto = summary_pb2.Summary()
    for key in dictionary:
        if dictionary[key] is None:
            continue
        if key == 'global_step':
            continue
        if (isinstance(dictionary[key], np.float32) or
                isinstance(dictionary[key], float)):
            summary_proto.value.add(
                tag=key, simple_value=float(dictionary[key]))
        elif (isinstance(dictionary[key], np.int64) or
              isinstance(dictionary[key], np.int32) or
              isinstance(dictionary[key], int)):
            summary_proto.value.add(tag=key, simple_value=int(dictionary[key]))
        elif isinstance(dictionary[key], six.binary_type):
            try:
                summ = summary_pb2.Summary.FromString(dictionary[key])
                for i, _ in enumerate(summ.value):
                    summ.value[i].tag = '%s/%d' % (key, i)
                summary_proto.value.extend(summ.value)
            except message.DecodeError:
                tf.logging.warn('Skipping summary for %s, '
                                'cannot parse string to Summary.',
                                key)
                continue
        else:
            tf.logging.warn(
                'Skipping summary for %s, must be a float, np.float32, '
                'np.int64, np.int32 or int or a serialized string of Summary.',
                key)
    summary_writer.add_summary(summary_proto, current_global_step)
    summary_writer.flush()


def _write_checkpoint_path_to_summary(output_dir, checkpoint_path,
                                      current_global_step):
    """Writes `checkpoint_path` into summary file in the given output
    directory.

    Args:
      output_dir: `str`, directory to write the summary file in.
      checkpoint_path: `str`, checkpoint file path to be written to summary
        file.
      current_global_step: `int`, the current global step.
    """

    checkpoint_path_tag = 'checkpoint_path'

    tf.logging.info('Saving \'%s\' summary for global step %d: %s',
                    checkpoint_path_tag, current_global_step, checkpoint_path)
    summary_proto = summary_pb2.Summary()
    summary_proto.value.add(
        tag=checkpoint_path_tag,
        tensor=tf.make_tensor_proto(
            checkpoint_path, dtype=tf.string))
    summary_writer = tf.summary.FileWriterCache.get(output_dir)
    summary_writer.add_summary(summary_proto, current_global_step)
    summary_writer.flush()


def _load_global_step_from_checkpoint_dir(checkpoint_dir):
    try:
        checkpoint_reader = tf.train.NewCheckpointReader(
            tf.train.latest_checkpoint(checkpoint_dir))
        return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
    except Exception:
        return 0
