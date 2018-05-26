#! /usr/bin/env python
# -*- coding=utf8 -*-

"""
From: https://stackoverflow.com/questions/45719176/how-to-display-runtime-statistics-in-tensorboard-using-estimator-api-in-a-distri
"""

import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs
from tensorflow.python.training import training_util
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer


class MetadataHook(SessionRunHook):

    def __init__(self, save_steps=None, output_dir=""):
        self.output_dir = output_dir
        self.timer = SecondOrStepTimer(every_secs=None, every_steps=save_steps)
        self.timer.update_last_triggered_step(0)

    def begin(self):
        self.next_step = 1
        self.global_step_tensor = training_util.get_global_step()
        self.writer = tf.summary.FileWriter(self.output_dir,
                                            tf.get_default_graph())

        if self.global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use ProfilerHook.")

    def before_run(self, run_context):
        self.request_summary = self.timer.should_trigger_for_step(
            self.next_step)
        requests = {"global_step": self.global_step_tensor}
        opts = (tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                if self.request_summary else None)
        return SessionRunArgs(requests, options=opts)

    def after_run(self, run_context, run_values):
        global_step = run_values.results["global_step"] + 1
        if self.request_summary:
            global_step = run_context.session.run(self.global_step_tensor)
            self.writer.add_run_metadata(
                run_values.run_metadata, "step-{}".format(global_step))
            self.writer.flush()
            self.timer.update_last_triggered_step(global_step)
        self.next_step = global_step + 1

    def end(self, session):
        self.writer.close()
