#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

import args_parser
import train
import input_data
import model_keys


def predict(opts):
    estimator = train.build_estimator(opts)
    tf.logging.info("Begin predict ...")
    if opts.preprocess_type == model_keys.PreprocessType.EASY:
        build_predict_input_fn = input_data.build_easy_predict_input_fn(opts)
    elif opts.preprocess_type == model_keys.PreprocessType.VGG:
        build_predict_input_fn = input_data.build_predict_input_fn(opts)
    else:
        raise ValueError("Unsurpported preprocess type.")

    results = estimator.predict(
        input_fn=build_predict_input_fn,
        checkpoint_path=opts.predict_checkpoint_path,
        yield_single_examples=True)
    with open(opts.predict_output, 'w') as fout, \
            open(opts.predict_data_path, 'r') as fin:
        for result in results:
            src = fin.readline().strip()
            fout.write(src + ' ')
            fout.write(str(result['score'][1]) + '\n')
    tf.logging.info("Predict done")


def main(argv):
    opts = args_parser.parse(argv)
    predict(opts)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
