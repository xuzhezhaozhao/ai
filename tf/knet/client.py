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


def run():
    channel = grpc.insecure_channel('localhost:9000')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = MODEL_NAME

    request.model_spec.signature_name = 'predicts'
    input_name = 'examples'

    x = ['' for i in range(20)]
    x[0] = '6055a60cdbf525ah'
    example1 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'words': _bytes_feature(x),
            }
        )
    ).SerializeToString()

    x = ['' for i in range(20)]
    x[0] = '9215a52047c920ae'
    example2 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'words': _bytes_feature(x),
            }
        )
    ).SerializeToString()

    examples = [example1, example2]
    request.inputs[input_name].CopyFrom(
        tf.contrib.util.make_tensor_proto(examples, dtype=tf.string))

    response = stub.Predict(request)
    print("Received: {}".format(response))


if __name__ == '__main__':
    run()
