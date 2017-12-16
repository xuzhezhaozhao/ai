#! /usr/bin/env python
# -*-coding:utf-8 -*-

from input_data_binary import read_data_sets_from_binary
from model import model_fn

import tensorflow as tf
import argparse
import sys
import numpy as np
import os


tf.logging.set_verbosity(tf.logging.INFO)

# Basic model parameters as external flags.
FLAGS = None


def serving_input_receiver_fn():
    """An input receiver that expects a serialized tf.Example."""
    feature_spec = {"watched":
                    tf.FixedLenFeature(dtype=tf.string,
                                       shape=[FLAGS.watched_size])}
    default_batch_size = None
    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                           shape=[default_batch_size],
                                           name='input_example_tensor')
    receiver_tensors = {'watched': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)

    # TODO convert rowkey to indices
    # watched_indices = tf.cast(features["watched"], tf.int64)
    # features = {"watched": watched_indices}

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def run_model():
    data_sets = read_data_sets_from_binary(
        FLAGS.train_watched_file,
        FLAGS.train_predicts_file,
        FLAGS.validation_watched_file,
        FLAGS.validation_predicts_file,
        FLAGS.test_watched_file,
        FLAGS.test_predicts_file,
        FLAGS.watched_size
    )

    # inputs = generate_average_inputs(video_embeddings, watched_pl)
    model_params = {
        "learning_rate": FLAGS.learning_rate,
        "embeddings_file_path": FLAGS.video_embeddings_file_binary,
        "num_sampled": FLAGS.num_sampled,
        "keep_prob": FLAGS.keep_prob,
        "k": FLAGS.k,
        "loss": FLAGS.loss,
        "embedding_dim": FLAGS.embedding_dim,
        "embeddings_trainable": FLAGS.embeddings_trainable
    }

    # Instantiate Estimator
    os.system("rm -rf " + FLAGS.model_dir)
    nn = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        params=model_params,
    )

    mode = FLAGS.run_mode
    # TODO magic 500000
    if mode == "train":
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"watched": data_sets.train.watched_videos[:-500000]},
            y=data_sets.train.predicts[:-500000],
            batch_size=FLAGS.batch_size,
            num_epochs=FLAGS.epoches,
            shuffle=True
        )
        nn.train(input_fn=train_input_fn)
    elif mode == "eval":
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"watched": data_sets.train.watched_videos[-500000:]},
            y=data_sets.train.predicts[-500000:],
            batch_size=FLAGS.batch_size,
            num_epochs=FLAGS.epoches,
            shuffle=True
        )
        nn.evaluate(input_fn=eval_input_fn)
    elif mode == "predict":
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            # x={"watched": data_sets.train.watched_videos[-2:]},
            x={"watched": np.array([[1,2,3,4,5,6,7,8,9,10]])},
            shuffle=False
        )
        predictions = nn.predict(input_fn=predict_input_fn)
        print(list(predictions))

    elif mode == "export":
        features = {'watched': tf.placeholder(tf.int64,
                                              [1, FLAGS.watched_size])}
        fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features)

        export_name = nn.export_savedmodel(
            export_dir_base=FLAGS.model_dir,
            serving_input_receiver_fn=fn,
        )
        print("export_name: {}".format(export_name))
    else:
        raise Exception("unkown run mode: {}".format(mode))


def main(_):
    run_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )

    parser.add_argument(
        '--epoches',
        type=int,
        default=20,
        help='Number of epoches to run trainer.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='Batch size.'
    )

    parser.add_argument(
        '--keep_prob',
        type=float,
        default=0.5,
        help='dropout keep probability.'
    )

    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=256,
        help='Word vector embedding dimension.'
    )

    parser.add_argument(
        '--num_sampled',
        type=int,
        default=10,
        help='number of sampled of NCE loss.'
    )

    parser.add_argument(
        '--watched_size',
        type=int,
        default=1,
        help="User's input watched size."
    )

    parser.add_argument(
        '--video_embeddings_file',
        type=str,
        default='',
        help='Pretrained video embeddings file in text format.'
    )

    parser.add_argument(
        '--video_embeddings_file_binary',
        type=str,
        default='',
        help='Pretrained video embeddings file in binary format.'
    )

    parser.add_argument(
        '--video_embeddings_file_dict',
        type=str,
        default='',
        help='Dict file for binary format pretrained video embeddings file.'
    )

    parser.add_argument(
        '--train_watched_file',
        type=str,
        default='',
        help='train watched data file.'
    )

    parser.add_argument(
        '--train_predicts_file',
        type=str,
        default='',
        help='train predicts data file.'
    )

    parser.add_argument(
        '--validation_watched_file',
        type=str,
        default='',
        help='validation watched data file.'
    )

    parser.add_argument(
        '--validation_predicts_file',
        type=str,
        default='',
        help='validation predicts data file.'
    )

    parser.add_argument(
        '--test_watched_file',
        type=str,
        default='',
        help='test watched data file.'
    )

    parser.add_argument(
        '--test_predicts_file',
        type=str,
        default='',
        help='test predicts data file.'
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        default='',
        help='Model dir.'
    )

    parser.add_argument(
        '--run_mode',
        type=str,
        default='train',
        help='Run mode - train, eval or predict'
    )

    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Predicts top k items.'
    )

    parser.add_argument(
        '--loss',
        type=str,
        default='nce',
        help='Loss funcition. [nce, softmax]'
    )

    parser.add_argument(
        '--embeddings_trainable',
        type=bool,
        default=True,
        help='Weather embeddings can be trainable.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
