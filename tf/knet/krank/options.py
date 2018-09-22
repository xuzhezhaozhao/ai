#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Options(object):
    def __init__(self):
        pass

    def list_all_member(self):
        print("\nOptions:")
        for name, value in vars(self).items():
            print('\t{} = {}'.format(name, value))
        print("\n")
