#!/bin/sh

python_out=python_out
cpp_out=cpp_out

mkdir -p ${python_out}

python -m grpc_tools.protoc -I. --python_out=${python_out} --grpc_python_out=${python_out} tensorflow_serving/apis/*

protoc -I. --grpc_out=${cpp_out} --plugin=protoc-gen-grpc=/usr/local/bin/grpc_cpp_plugin tensorflow_serving/apis/*
protoc -I. --cpp_out=${cpp_out} tensorflow_serving/apis/*

protoc -I. --cpp_out=${cpp_out} tensorflow/core/framework/*
