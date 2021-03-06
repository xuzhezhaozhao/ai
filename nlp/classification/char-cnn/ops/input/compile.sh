#! /usr/bin/env bash

set -e

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${MYDIR}

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
echo "TF_CFLAGS = " ${TF_CFLAGS[@]}
echo "TF_LFLAGS = " ${TF_LFLAGS[@]}

echo "compile char_cnn_input_ops.so ..."
g++ -std=c++11 -shared \
    char_cnn_input_ops.cc \
    char_cnn_input_kernels.cc \
    -o char_cnn_input_ops.so \
    -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
