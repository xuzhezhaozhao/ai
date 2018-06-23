#! /usr/bin/env bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
echo "TF_CFLAGS = " ${TF_CFLAGS[@]}
echo "TF_LFLAGS = " ${TF_LFLAGS[@]}
echo "compile tfrecord writer ..."

g++ -std=c++11 \
    tfrecord_writer.cc \
    dictionary.cc \
    -o tfrecord_writer \
    ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -lgflags -O2
