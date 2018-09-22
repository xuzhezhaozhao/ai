#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import grpc
import tensorflow as tf
import tensorflow_serving.apis.predict_pb2 as predict_pb2
import tensorflow_serving.apis.prediction_service_pb2_grpc as prediction_service_pb2_grpc


MODEL_NAME = "knet"


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def run():
    channel = grpc.insecure_channel('localhost:9000')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = MODEL_NAME

    request.model_spec.signature_name = 'predicts'
    input_name = 'examples'

    watched_rowkeys = ['' for i in range(100)]
    watched_rowkeys[0] = '8575b81e0d8430aj'
    rinfo1 = [10.0 for i in range(100)]
    rinfo2 = [10.0 for i in range(100)]
    is_video = [1 for i in range(100)]
    target_rowkeys = ['' for i in range(200)]
    target_rowkeys[0] = '8575b81e0d8430aj'

    example1 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'watched_rowkeys': _bytes_feature(watched_rowkeys),
                'rinfo1': _float_feature(rinfo1),
                'rinfo2': _float_feature(rinfo2),
                'target_rowkeys': _bytes_feature(target_rowkeys),
                'is_video': _int64_feature(is_video)
            }
        )
    ).SerializeToString()

    examples = [example1]
    request.inputs[input_name].CopyFrom(
        tf.contrib.util.make_tensor_proto(examples, dtype=tf.string))

    response = stub.Predict(request)
    print("Received: {}".format(response))


if __name__ == '__main__':
    run()
