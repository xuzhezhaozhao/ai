#! /usr/bin/env python
# -*- coding=utf8 -*-

"""
ref:
https://stackoverflow.com/questions/45719176/how-to-display-runtime-statistics-in-tensorboard-using-estimator-api-in-a-distri
"""

import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training import training_util


class MetadataHook(SessionRunHook):

    def __init__(self, save_steps=None, output_dir=""):
        self.output_dir = output_dir
        self.save_steps = save_steps

    def begin(self):
        self.writer = tf.summary.FileWriter(self.output_dir,
                                            tf.get_default_graph())

    def before_run(self, run_context):
        self.global_step_tensor = training_util.get_global_step()
        self.global_step = run_context.session.run(self.global_step_tensor)

        if self.global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use MetadataHook.")

        self.should_summary = False
        opts = None
        next_step = self.global_step + 1
        if next_step % self.save_steps == 0:
            opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.should_summary = True

        fetches = {}
        return SessionRunArgs(fetches, options=opts)

    def after_run(self, run_context, run_values):
        self.global_step += 1
        if self.should_summary:
            self.writer.add_run_metadata(run_values.run_metadata,
                                         str(self.global_step))
            self.writer.flush()

    def end(self, session):
        self.writer.close()
