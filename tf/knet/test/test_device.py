
# try running cpu intensive test on two devices

import tensorflow as tf
import time

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


def matmul_op():
    """Multiply two matrices together"""

    n = 2000
    a = tf.ones((n, n), dtype=tf.float32)
    return tf.matmul(a, a) / n


slow_op = matmul_op

with tf.device("/cpu:0"):
    one = slow_op()
with tf.device("/cpu:0"):
    another_one = slow_op()

config = tf.ConfigProto(device_count={"CPU": 2},
                        inter_op_parallelism_threads=4,
                        intra_op_parallelism_threads=4)
config.graph_options.optimizer_options.opt_level = -1

sess = tf.Session(config=config)

start = time.time()
sess.run([one, another_one])
elapsed_time2 = time.time() - start
print("Two ops in parallel: %.2f sec" % (elapsed_time2))

start = time.time()
sess.run(one)
elapsed_time = time.time() - start
print("Single op: %2.4f sec (%.2f times faster)" %
      (elapsed_time, elapsed_time2 / elapsed_time))
