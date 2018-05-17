
import tensorflow as tf
import argparse
import traceback
import sys


FLAGS = None

string_split_v2_module = None
string_split_v2 = None


def main():
    filenames = [FLAGS.input]
    # TextLineDataset 是流式的, 可以处理大规模数据
    dataset = tf.data.TextLineDataset(filenames)
    dataset = dataset.map(lambda x: string_split_v2(x, FLAGS.delimter))
    dataset = dataset.padded_batch(5, padded_shapes=[None])
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

    parser.add_argument(
        "--delimter",
        type=str,
        default=" ",
        help=""
    )

    parser.add_argument(
        "--string_split_v2_ops_path",
        type=str,
        default="",
        help=""
    )

    FLAGS, unparsed = parser.parse_known_args()
    string_split_v2_module = tf.load_op_library(FLAGS.string_split_v2_ops_path)
    string_split_v2 = string_split_v2_module.string_split_v2

    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
        sys.exit(-1)
