
from __future__ import print_function

import grpc

import tensorflow as tf
import tensorflow_serving.apis.predict_pb2 as predict_pb2
import tensorflow_serving.apis.prediction_service_pb2_grpc as prediction_service_pb2_grpc


MODEL_NAME = "knet"


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def run():
    channel = grpc.insecure_channel('localhost:9000')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = MODEL_NAME

    request.model_spec.signature_name = 'predicts'
    input_name = 'examples'

    example1 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'records': _int64_feature([11, 22, 23, 34, 55]),
            }
        )
    ).SerializeToString()

    example2 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'records': _int64_feature([1, 2, 3, 4, 5]),
            }
        )
    ).SerializeToString()

    example3 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'records': _int64_feature([100, 200, 300, 400, 500]),
            }
        )
    ).SerializeToString()

    examples = [example1, example2, example3]
    request.inputs[input_name].CopyFrom(
        tf.contrib.util.make_tensor_proto(examples, dtype=tf.string))

    response = stub.Predict(request)
    print("Received: {}".format(response))


if __name__ == '__main__':
    run()
