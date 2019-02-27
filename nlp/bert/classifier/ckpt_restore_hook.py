
"""Copied from: https://github.com/JayYip/bert-multitask-learning/blob/master/src/ckpt_restore_hook.py"""

from bert import modeling
import tensorflow as tf


class RestoreCheckpointHook(tf.train.SessionRunHook):
    def __init__(self, checkpoint_path):
        tf.logging.info("Create RestoreCheckpointHook.")
        self.checkpoint_path = checkpoint_path

    def begin(self):
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if self.checkpoint_path:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(
                tvars, self.checkpoint_path)
            tf.train.init_from_checkpoint(self.checkpoint_path, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

    def after_create_session(self, session, coord):
        pass

    def before_run(self, run_context):
        return None

    def after_run(self, run_context, run_values):
        pass

    def end(self, session):
        pass
