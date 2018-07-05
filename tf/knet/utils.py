#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


def normalize_matrix(x):
    norm = np.linalg.norm(x, axis=1)
    x = x / norm.reshape([-1, 1])
    return x
