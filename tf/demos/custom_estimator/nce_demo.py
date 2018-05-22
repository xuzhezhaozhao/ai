#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a custom Estimator for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import os
import math

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

DATA_DIR = "../test_data/"


def parse_csv(x):
    CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]
    fields = tf.decode_csv(x, CSV_TYPES)
    features = dict(zip(CSV_COLUMN_NAMES, fields))
    label = features.pop('Species')
    return (features, label)


def train_input_fn(filename, skip_rows=0):
    ds = tf.data.TextLineDataset(filename).skip(skip_rows)
    ds = ds.map(parse_csv, num_parallel_calls=4)
    ds = ds.shuffle(1000).repeat().batch(100)
    return ds


def eval_input_fn(filename, skip_rows=0):
    ds = tf.data.TextLineDataset(filename).skip(skip_rows)
    ds = ds.map(parse_csv)
    ds = ds.batch(100)
    return ds


def predict_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
args = None


def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    n_classes = params['n_classes']
    embedding_dim = 32

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([n_classes, embedding_dim],
                            stddev=1.0 / math.sqrt(embedding_dim)))
    nce_biases = tf.Variable(tf.zeros([n_classes]))

    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.

    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    net = tf.layers.dense(net, embedding_dim, activation=None)
    logits = tf.matmul(net, tf.transpose(nce_weights))
    logits = tf.nn.bias_add(logits, nce_biases)

    # Compute predictions.
    serving_default = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        export_outputs = {
            serving_default: tf.estimator.export.ClassificationOutput(
                scores=tf.nn.softmax(logits),
                classes=tf.constant(SPECIES, dtype=tf.string)
            ),
            'predicts': tf.estimator.export.PredictOutput(
                outputs={
                    'class_ids': predicted_classes[:, tf.newaxis],
                    'probabilities': tf.nn.softmax(logits),
                    'logits': logits
                }
            )
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions,
                                          export_outputs=export_outputs)

    # Compute nce_loss.
    nce_loss = tf.nn.nce_loss(weights=nce_weights,
                              biases=nce_biases,
                              labels=tf.reshape(labels, [-1, 1]),
                              inputs=net,
                              num_sampled=2,
                              num_classes=n_classes)
    nce_loss = tf.reduce_mean(nce_loss)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(nce_loss,
                                  global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=nce_loss, train_op=train_op)


def feature_default():
    return tf.FixedLenFeature(shape=[1], dtype=tf.float32, default_value=0.0)


feature_spec = {
    'SepalLength': feature_default(),
    'SepalWidth': feature_default(),
    'PetalLength': feature_default(),
    'PetalWidth': feature_default()
}


def serving_input_receiver_fn():
    """An input receiver that expects a serialized tf.Example.
    Note: Set serialized_tf_example shape as [None] to handle variable
    batch size
    """
    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                           shape=[None],
                                           name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}
    raw_features = tf.parse_example(serialized_tf_example, feature_spec)

    features = raw_features

    # Do anything to raw_features ...
    # such as
    # features = {
    #     'SepalLength': tf.constant([0.0, 0.0]),
    #     'SepalWidth': tf.constant([0.0, 0.0]),
    #     'PetalLength': tf.constant([0.0, 0.0]),
    #     'PetalWidth': tf.constant([0.0, 0.0])
    # }
    # or
    # features = {}
    # for key in raw_features.keys():
    #     features[key] = tf.constant([0.0, 0.0])

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def main(argv):
    args = parser.parse_args(argv[1:])

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in CSV_COLUMN_NAMES[:-1]:
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10, 10],
            # The model must choose between 3 classes.
            'n_classes': 3,
        })

    # Train the Model.
    classifier.train(
        input_fn=lambda: train_input_fn(
            os.path.join(DATA_DIR, 'iris_training.csv'), 1),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(
            os.path.join(DATA_DIR, 'iris_test.csv'), 1))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda: predict_input_fn(predict_x,
                                          labels=None,
                                          batch_size=args.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(SPECIES[class_id], 100 * probability, expec))

    classifier.export_savedmodel(
        "model_dir",
        serving_input_receiver_fn=serving_input_receiver_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
