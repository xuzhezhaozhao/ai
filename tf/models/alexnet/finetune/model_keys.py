#! /usr/bin/env python
# -*- coding=utf8 -*-


DATA_COL = 'data'


class TaskType(object):
    """Distribute task type."""

    LOCAL = 'local'  # non-distributed
    CHIEF = 'chief'
    WORKER = 'worker'
    PS = 'ps'
    EVALUATOR = 'evaluator'


class OptimizerType(object):
    """Optimizer type."""

    ADA = 'ada'
    ADADELTA = 'adadelta'
    ADAM = 'adam'
    SGD = 'sgd'
    RMSPROP = 'rmsprop'


class TrainParallelMode(object):
    """Train parallen mode."""

    DEFAULT = 'default'
    TRAIN_OP_PARALLEL = 'train_op_parallel'
    MULTI_THREAD = 'multi_thread'
    MULTI_THREAD_V2 = 'multi_thread_v2'


class SGDLrDecayType(object):
    """SGD learning rate decay type."""

    NONE = 'none'
    EXPONENTIAL_DECAY = 'exponential_decay'
    FASTTEXT_DECAY = 'fasttext_decay'
    POLYNOMIAL_DECAY = 'polynomial_decay'
