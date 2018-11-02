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


class PreprocessType(object):
    EASY = 'easy'
    VGG = 'vgg'
