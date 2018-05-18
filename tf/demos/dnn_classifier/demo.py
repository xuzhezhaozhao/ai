
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


def train_decode_line(x):
    tensors = tf.decode_csv(x, [[] for _ in range(len(CSV_COLUMN_NAMES))])
    tensors = map(lambda x: tf.reshape(x, [-1]), tensors)
    label = tf.cast(tensors[-1], tf.int32)
    features = dict(zip(CSV_COLUMN_NAMES[:-1], tensors[:-1]))
    return (features, label)


def train_input_fn(filename, skip_rows=0):
    ds = tf.data.TextLineDataset(filename).skip(skip_rows)
    ds = ds.map(train_decode_line)
    ds = ds.shuffle(1000).repeat().batch(100)
    return ds


def eval_input_fn(filename, skip_rows=0):
    ds = tf.data.TextLineDataset(filename).skip(skip_rows)
    ds = ds.map(train_decode_line)
    ds = ds.batch(100)
    return ds


def main(argv):
    my_feature_colums = []
    for key in CSV_COLUMN_NAMES[:-1]:
        my_feature_colums.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_colums,
        hidden_units=[10, 10, 10],
        n_classes=3
    )
    classifier.train(
        input_fn=lambda: train_input_fn('iris_training.csv', 1), steps=1000)
    train_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn('iris_training.csv', 1))
    print('\nTraining set accuracy: {accuracy:0.3f}\n'.format(**train_result))

    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn('iris_test.csv', 1))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
