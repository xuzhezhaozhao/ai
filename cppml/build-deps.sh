#! /usr/bin/env bash

echo "building deps/googletest ..."
pre_dir=`pwd`
cd deps/googletest && mkdir -p build && cd build && cmake .. && make
cd $pre_dir
