#! /usr/bin/env bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
echo "TF_CFLAGS = " ${TF_CFLAGS[@]}
echo "TF_LFLAGS = " ${TF_LFLAGS[@]}

echo "compile text_cnn_input_ops.so ..."
g++ -std=c++11 -shared \
    text_cnn_input_ops.cc \
    text_cnn_input_kernels.cc \
    -o text_cnn_input_ops.so \
    -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
