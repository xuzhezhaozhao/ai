#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

import args_parser
import train
import input_data


def predict(opts):
    estimator = train.build_estimator(opts)
    tf.logging.info("Begin predict ...")
    if opts.use_easy_preprocess:
        predict_input_fn = input_data.build_easy_predict_input_fn(opts)
    else:
        predict_input_fn = input_data.build_predict_input_fn(opts)

    results = estimator.predict(input_fn=predict_input_fn)
    with open(opts.predict_output, 'w') as fout, \
            open(opts.predict_data_path, 'r') as fin:
        for result in results:
            src = fin.readline().strip()
            fout.write(src + ' ')
            fout.write(str(result['score'][0]) + '\n')
    tf.logging.info("Predict done")


def main(argv):
    opts = args_parser.parse(argv)
    predict(opts)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
