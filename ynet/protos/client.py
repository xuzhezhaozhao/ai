
from __future__ import print_function

import grpc

import tensorflow as tf
import tensorflow_serving.apis.predict_pb2 as predict_pb2
import tensorflow_serving.apis.prediction_service_pb2_grpc as prediction_service_pb2_grpc


def run():
    channel = grpc.insecure_channel('localhost:9000')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'default'
    request.model_spec.signature_name = 'predicts'
    request.inputs['watched'].CopyFrom(
        tf.contrib.util.make_tensor_proto(range(5),
                                          dtype=tf.int64,
                                          shape=[1, 5])
    )

    response = stub.Predict(request)
    print("Received: {}".format(response))


if __name__ == '__main__':
    run()
