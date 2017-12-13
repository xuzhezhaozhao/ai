#!/bin/sh

python_out=python_out

mkdir -p ${python_out}

python -m grpc_tools.protoc -I. --python_out=${python_out} --grpc_python_out=${python_out} tensorflow_serving/apis/*
#python -m grpc_tools.protoc -I. --python_out=${python_out} --grpc_python_out=${python_out} tensorflow/core/example/*
#python -m grpc_tools.protoc -I. --python_out=${python_out} --grpc_python_out=${python_out} tensorflow/core/framework/*
#python -m grpc_tools.protoc -I. --python_out=${python_out} --grpc_python_out=${python_out} tensorflow/core/protobuf/*
