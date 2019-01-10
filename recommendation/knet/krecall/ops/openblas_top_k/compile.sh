#! /usr/bin/env bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
echo "TF_CFLAGS = " ${TF_CFLAGS[@]}
echo "TF_LFLAGS = " ${TF_LFLAGS[@]}
echo "compile openblas_top_k_ops ..."

g++ -std=c++11 -shared \
    openblas_top_k_ops.cc \
    openblas_top_k_kernels.cc \
    openblas_util.cc \
    vector.cc \
    matrix.cc \
    -o openblas_top_k_ops.so \
    -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
