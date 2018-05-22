#!/bin/sh

python_out=python_out

mkdir -p ${python_out}

python -m grpc_tools.protoc -I. --python_out=${python_out} --grpc_python_out=${python_out} tensorflow_serving/apis/*
touch ${python_out}/tensorflow_serving/__init__.py
touch ${python_out}/tensorflow_serving/apis/__init__.py
cd ${python_out}
ln -sf ../../client.py client.py
cd ..
