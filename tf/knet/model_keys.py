#! /usr/bin/env python
# -*- coding=utf8 -*-

PADDING_ID = 0

NCE_WEIGHTS_NAME = 'nce_weights'
NCE_BIASES_NAME = 'nce_biases'

# filename must end with .npy
SAVE_NCE_WEIGHTS_NAME = 'nce_weights.npy'
SAVE_NCE_BIASES_NAME = 'nce_biases.npy'
SAVE_NCE_WEIGHTS_SUBSET_NAME = 'nce_weights_subset.npy'
SAVE_NCE_BIASES_SUBSET_NAME = 'nce_biases_subset.npy'

NCE_PARAM_NAMES = [
    SAVE_NCE_WEIGHTS_NAME,
    SAVE_NCE_BIASES_NAME,
    SAVE_NCE_WEIGHTS_SUBSET_NAME,
    SAVE_NCE_BIASES_SUBSET_NAME,
]

OPTIMIZE_LEVEL_ZERO = 0
OPTIMIZE_LEVEL_SAVED_NCE_PARAMS = 1
OPTIMIZE_LEVEL_OPENBLAS_TOP_K = 2

ALL_OPTIMIZE_LEVELS = [
    OPTIMIZE_LEVEL_ZERO,
    OPTIMIZE_LEVEL_SAVED_NCE_PARAMS,
    OPTIMIZE_LEVEL_OPENBLAS_TOP_K,
]


RECORDS_COL = 'records'
WORDS_COL = 'words'
NUM_IN_DICT_COL = 'num_in_dict'
TOKENS_COL = 'tokens'

DICT_META = 'dict_meta'
DICT_WORDS = 'dict_words'
SAVED_DICT_BIN = 'saved_dict.bin'
DICT_WORDS_SUBSET = 'dict_words_subset'

DICT_PARAM_NAMES = [
    DICT_META,
    DICT_WORDS,
    SAVED_DICT_BIN,
    DICT_WORDS_SUBSET,
]


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


class TrainDataFormatType(object):
    """Train data format type."""

    fasttext = 'fasttext'
    tfrecord = 'tfrecord'


class TrainParallelMode(object):
    """Train parallen mode."""

    DEFAULT = 'default'
    TRAIN_OP_PARALLEL = 'train_op_parallel'


class SGDLrDecayType(object):
    """SGD learning rate decay type."""

    NONE = 'none'
    EXPONENTIAL_DECAY = 'exponential_decay'
    FASTTEXT_DECAY = 'fasttext_decay'
