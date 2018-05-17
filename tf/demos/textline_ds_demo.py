
import tensorflow as tf
import numpy as np
import argparse
import traceback
import sys


FLAGS = None


def main():
    filenames = [FLAGS.input]
    dataset = tf.data.TextLineDataset(filenames)

    dataset = dataset.repeat(FLAGS.epoch)
    iterator = dataset.make_one_shot_iterator()

    with tf.Session() as sess:
        for i in range(20):
            try:
                line = sess.run(iterator.get_next())
                print(line)
            except tf.errors.OutOfRangeError:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help=""
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
        help=""
    )

    FLAGS, unparsed = parser.parse_known_args()
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
        sys.exit(-1)
