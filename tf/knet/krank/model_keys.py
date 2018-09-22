#! /usr/bin/env python
# -*- coding=utf8 -*-

PADDING_ID = 0


POSITIVE_RECORDS_COL = 'positive_records'
NEGATIVE_RECORDS_COL = 'negative_records'
TARGETS_COL = 'targets'


class TrainParallelMode(object):
    """Train parallen mode."""

    DEFAULT = 'default'
    TRAIN_OP_PARALLEL = 'train_op_parallel'
    MULTI_THREAD = 'multi_thread'
    MULTI_THREAD_V2 = 'multi_thread_v2'


class OptimizerType(object):
    """Optimizer type."""

    ADA = 'ada'
    ADADELTA = 'adadelta'
    ADAM = 'adam'
    SGD = 'sgd'
    RMSPROP = 'rmsprop'
