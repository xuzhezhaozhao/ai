#! /usr/bin/env python
# -*- coding=utf8 -*-

PADDING_ID = 0


POSITIVE_RECORDS_COL = 'positive_records'
NEGATIVE_RECORDS_COL = 'negative_records'
TARGETS_COL = 'targets'


class TrainParallelMode(object):
    """Train parallen mode."""

    DEFAULT = 'default'
    MULTI_THREAD = 'multi_thread'


class OptimizerType(object):
    """Optimizer type."""

    ADAGRAD = 'adagrad'
    ADADELTA = 'adadelta'
    ADAM = 'adam'
    SGD = 'sgd'
    RMSPROP = 'rmsprop'
    MOMENTUM = 'momentum'
    FTRL = 'ftrl'


class LossType(object):
    """Loss type."""

    CE = 'ce'
    MSE = 'mse'
