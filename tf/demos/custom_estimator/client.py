
from __future__ import print_function

import grpc

import tensorflow as tf
import tensorflow_serving.apis.predict_pb2 as predict_pb2
import tensorflow_serving.apis.prediction_service_pb2_grpc as prediction_service_pb2_grpc


MODEL_NAME = "custom_estimator"


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def run():
    channel = grpc.insecure_channel('localhost:9000')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = MODEL_NAME

    # request.model_spec.signature_name = 'predicts'
    # input_name = 'examples'

    # model have another export signature_name, try follows
    serving_default = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    request.model_spec.signature_name = serving_default
    input_name = 'inputs'

    example1 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'SepalLength': _float32_feature([5.1]),
                'SepalWidth': _float32_feature([3.3]),
                'PetalLength': _float32_feature([1.7]),
                'PetalWidth': _float32_feature([0.5]),
            }
        )
    ).SerializeToString()

    example2 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'SepalLength': _float32_feature([5.9]),
                'SepalWidth': _float32_feature([3.0]),
                'PetalLength': _float32_feature([4.2]),
                'PetalWidth': _float32_feature([1.5]),
            }
        )
    ).SerializeToString()

    example3 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'SepalLength': _float32_feature([6.9]),
                'SepalWidth': _float32_feature([3.1]),
                'PetalLength': _float32_feature([5.4]),
                'PetalWidth': _float32_feature([2.1]),
            }
        )
    ).SerializeToString()

    request.inputs[input_name].CopyFrom(
        tf.contrib.util.make_tensor_proto([example1, example2, example3],
                                          dtype=tf.string))

    response = stub.Predict(request)
    print("Received: {}".format(response))


if __name__ == '__main__':
    run()
