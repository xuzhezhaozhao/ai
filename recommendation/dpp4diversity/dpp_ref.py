import datetime
import os

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import argparse
import time
import requests
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2


def parse_args():
    # python3 train.py --train_file
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocabs_dir", type=str, default="vocabs/", help="Base directory for the model.")
    parser.add_argument("--model_dir", type=str, default="model/", help="Base directory for the model.")
    parser.add_argument("--export_dir", type=str, default="saved_model", help="Base directory to export model.")
    parser.add_argument("--max_length", type=int, default=100, help="return max length")
    parser.add_argument("--monitor_model_name", type=str, default="tensorflow model", help="model name for monitor")
    parser.add_argument("--monitor_qywx", type=str, default="samsonqi;tivonyan;", help="notify the trainning result to")
    parser.add_argument("--theta", type=float, default=0.8, help="trade-off of diversity and relative")
    parser.add_argument("--dict", type=str, default="hbcf", help="source of dict")
    parser.add_argument("--ftime", type=str, default="20190813", help="ftime")
    parser.add_argument("--dim", type=int, default=200, help="embedding size")
    parser.add_argument("--padding", type=int, default=200, help="padding num")
    parser.add_argument("--vocab_size", type=int, default=0, help="vocab size")

    return parser.parse_known_args()

FLAGS, unparsed = parse_args()

# process dssm dict
def preprocess_dssm(vocab, rowkey_file, embedding_file, skip_rows=0, sep=' ', dim=200, padding=200, keep_rowkeys_file=None):
    print("preprocessing... %s:%s:%s" % (vocab, rowkey_file, embedding_file))
    row_num = 0
    with open(vocab, encoding="utf8",  newline="\n") as fi, \
            open(rowkey_file, encoding="utf8", mode="w", newline="\n") as fr, \
            open(embedding_file, encoding="utf8", mode="w", newline="\n") as fe:
        rowkeys = set()
        
        if keep_rowkeys_file is not None:
            with open(keep_rowkeys_file, encoding="utf8", newline="\n") as f:
                rowkeys = set(r.strip("\n") for r in f.readlines())
                print("rowkeys: %d" % len(rowkeys))
        for line in fi:
            if row_num < skip_rows:
                row_num += 1
                continue
            rowkey, embedding = line.split(sep, 1)
            if keep_rowkeys_file is not None:
                if rowkey in rowkeys:
                    fr.write(rowkey + "\n")
                    fe.write(embedding.strip() + "\n")
                    row_num += 1
            else:
                fr.write(rowkey + "\n")
                fe.write(embedding.strip() + "\n")
                row_num += 1
        if padding > 0:
            for i in range(padding):
                fr.write('UNK_' + str(i) + "\n")
                fe.write(','.join(['0.1' if i == j else '0.0' for j in range(dim)]) + "\n")
    return row_num - skip_rows


def dpp(kernel_matrix, max_length, epsilon=1E-10):
    di2s = tf.identity(tf.diag_part(kernel_matrix))
    selected_items = []
    selected_item = tf.argmax(di2s)
    selected_items.append(selected_item)
    cis = []
    items_size = tf.shape(di2s)[0]

    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        di_optimal = tf.math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        if k > 0:
            ci_optimal = tf.reshape(tf.gather(cis[:k], selected_item, axis=1), [-1, k])
            x = tf.stack(cis[:k])
            eis = tf.squeeze((elements - tf.matmul(ci_optimal, x)) / di_optimal)
        else:
            eis = tf.squeeze(elements / di_optimal)

        cis.append(eis)
        di2s = tf.subtract(di2s, tf.square(eis))
        selected_item = tf.argmax(di2s)

        def true_fn():
            selected_items.append(selected_item)
            return max_length

        def false_fn():
            max_length = len(selected_items)
            return max_length
        tf.cond(di2s[selected_item] < epsilon, true_fn, false_fn)
    truncate_size = tf.cond(items_size < max_length, lambda : items_size, lambda : max_length)
    return tf.squeeze(tf.slice([tf.stack(selected_items)], [0, 0], [-1, truncate_size]))


def signature(function_dict):
    signature_dict = {}
    for k, v in function_dict.items():
        inputs = {k: tf.saved_model.utils.build_tensor_info(v) for k, v in v['inputs'].items()}
        outputs = {k: tf.saved_model.utils.build_tensor_info(v) for k, v in v['outputs'].items()}
        signature_dict[k] = tf.saved_model.build_signature_def(inputs=inputs, outputs=outputs, method_name=v['method_name'])
    return signature_dict


def export(model_dir, vocab_dir, max_length, theta, emb_size, dim, num_oov_buckets):
    while True:
        cur = os.path.join(model_dir, str(int(time.time())))
        if not tf.gfile.Exists(cur):
            break
    print("export model path: %s" % cur)
    method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME

    rowkeys_file = os.path.join(vocab_dir, "rowkeys.vocab")
    embedding_file = os.path.join(vocab_dir, "emb.vocab")

    with tf.Graph().as_default(), tf.Session() as sess:
        rowkeys = tf.placeholder(dtype=tf.string, name="rowkeys")
        algorithm_ids = tf.placeholder(dtype=tf.uint32, name="algorithm_id")
        scores = tf.placeholder(dtype=tf.float32, name="scores")


        # rowkeys = tf.constant(["2785c0f6f0e592ah",
        #                        "8605d35a21f857bk",
        #                        "7915d39c6f9755ap",
        #                        "3155d3846cb468bk",
        #                        "4285d39597b375bk"], dtype=tf.string)
        # algorithm_ids = tf.constant([2081,
        #                             2803,
        #                             2086,
        #                             2803,
        #                             2086], dtype=tf.uint32)
        # scores = tf.constant([
        #                                     0.1,
        #                                     0.2,
        #                                     0.3,
        #                                     0.11,
        #                                     0.7
        #                                 ], dtype=float)

        with open(rowkeys_file, encoding="utf8") as fi:
            lines = fi.readlines()
            vocab_size = len(lines)
            print(vocab_size)

        emb = tf.Variable(np.loadtxt(embedding_file, delimiter=' '), dtype=tf.float32)

        print(emb.shape)

        table = tf.contrib.lookup.index_table_from_file(vocabulary_file=rowkeys_file,
                                                        vocab_size=vocab_size-num_oov_buckets,
                                                        hasher_spec=tf.contrib.lookup.FastHashSpec,
                                                        num_oov_buckets=num_oov_buckets)
        rowkey_ids = table.lookup(rowkeys)
        rowkeys_embedding = tf.nn.embedding_lookup(emb, rowkey_ids)
        rowkeys_embedding /= tf.linalg.norm(rowkeys_embedding, axis=1, keepdims=True)
        rowkeys_embedding = tf.where(tf.is_nan(rowkeys_embedding), tf.zeros_like(rowkeys_embedding), rowkeys_embedding)
        similarities = tf.cast(tf.matmul(rowkeys_embedding, rowkeys_embedding, transpose_b=True), tf.float32)
        kernel_matrix = tf.reshape(scores, [-1, 1]) * similarities * tf.reshape(scores, [1, -1])
        # alpha = theta / (2 * (1 - theta))
        # kernel_matrix = tf.math.exp(alpha * tf.reshape(scores, [-1, 1])) * similarities * tf.math.exp(alpha * tf.reshape(scores, [1, -1]))

        indices = dpp(kernel_matrix, max_length)

        predict_rowkeys = tf.gather(rowkeys, indices)
        predict_scores = tf.gather(scores, indices)
        predict_algorithm_ids = tf.gather(algorithm_ids, indices)
        predict_positions = indices

        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        #sess.run(emb, feed_dict={embedding_placeholder: emb_dict})
        # print(sess.run(predict_rowkeys))
        # print(sess.run(predict_scores))
        signature_def_map = signature({
            "prediction": {
                "inputs": {
                    'rowkeys': rowkeys,
                    "scores": scores,
                    "algorithm_ids": algorithm_ids
                },
                "outputs": {
                    "rowkeys": predict_rowkeys,
                    "scores": predict_scores,
                    "algorithm_ids": predict_algorithm_ids,
                    "origin_position": predict_positions
                },
                "method_name": method_name
            },
        })

        builder = tf.saved_model.builder.SavedModelBuilder(cur)
        builder.add_meta_graph_and_variables(sess, tags=[tf.saved_model.tag_constants.SERVING],
                                             signature_def_map=signature_def_map,
                                             assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
                                             main_op=tf.tables_initializer())
        builder.save()

        os.mkdir(os.path.join(cur, "assets.extra"))
        with tf.python_io.TFRecordWriter(os.path.join(cur, "assets.extra/tf_serving_warmup_requests")) as writer:
            request = predict_pb2.PredictRequest(
                model_spec=model_pb2.ModelSpec(name="kva_dpp_model", signature_name="prediction"),
                inputs={
                    "rowkeys": tf.make_tensor_proto(["2785c0f6f0e592ah", "2785c0f6f0e592ah"], dtype=tf.string),
                    "algorithm_ids": tf.make_tensor_proto([2081, 2081], dtype=tf.uint32),
                    "scores": tf.make_tensor_proto([0.7, 0.7], dtype=tf.float32)
                }
            )
            print(request)
            log = prediction_log_pb2.PredictionLog(
                predict_log=prediction_log_pb2.PredictLog(request=request))
            writer.write(log.SerializeToString())


def monitor(reciever, title, text):
    data = '{"receiver":"%s","msg":"%s", "title":"%s"}' % (reciever, text, title)
    rsp = requests.post(url='http://t.isd.com/api/sendQiYeWX', data=data)
    return rsp

if __name__ == '__main__':
    #raw_rowkeys = os.path.join(FLAGS.vocabs_dir, 'raw_rowkeys.vocab')
    #dssm_rowkeys = os.path.join(FLAGS.vocabs_dir, 'rowkeys.vocab')
    #dssm_embeddings = os.path.join(FLAGS.vocabs_dir, 'emb.vocab')
    #dssm_vocab = os.path.join(FLAGS.vocabs_dir, 'video_embeddings_' + FLAGS.ftime)
                              #+ (datetime.date.today() + datetime.timedelta(days=-1)).strftime('%Y%m%d'))
    #dssm_vocab_size = preprocess_dssm(dssm_vocab, dssm_rowkeys, dssm_embeddings, skip_rows=0, sep='||',
    #                                  dim=200, padding=200, keep_rowkeys_file=raw_rowkeys)
    if FLAGS.vocab_size >= 100000:
        export(FLAGS.model_dir, FLAGS.vocabs_dir, FLAGS.max_length, FLAGS.theta, FLAGS.vocab_size, FLAGS.dim, FLAGS.padding)
        monitor(FLAGS.monitor_qywx, 'dpp tf serving model notification', '%s dpp model generate successfully: %d' % (FLAGS.ftime, FLAGS.vocab_size))
    else:
        monitor(FLAGS.monitor_qywx, 'dpp tf serving model notification', "%s vocab size exception: %d" % (FLAGS.ftime, FLAGS.vocab_size))
