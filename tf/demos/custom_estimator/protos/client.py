
from __future__ import print_function

import grpc

import tensorflow as tf
import tensorflow_serving.apis.predict_pb2 as predict_pb2
import tensorflow_serving.apis.prediction_service_pb2_grpc as prediction_service_pb2_grpc


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def run():
    channel = grpc.insecure_channel('localhost:9000')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'default'
    request.model_spec.signature_name = 'predicts'

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'SepalLength': _float32_feature([5.1]),
                'SepalWidth': _float32_feature([5.1]),
                'PetalLength': _float32_feature([5.1]),
                'PetalWidth': _float32_feature([5.1]),
            }
        )
    ).SerializeToString()

    request.inputs['examples'].CopyFrom(
        tf.contrib.util.make_tensor_proto([example],
                                          dtype=tf.string,
                                          shape=[1])
    )

    response = stub.Predict(request)
    print("Received: {}".format(response))


if __name__ == '__main__':
    run()
