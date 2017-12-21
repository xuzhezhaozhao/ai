project(preprocess)
cmake_minimum_required(VERSION 2.8)

find_package(yaaf-ng REQUIRED)
find_package(extlib REQUIRED)

set(CMAKE_CXX_FLAGS "-g -O2 -Wall -std=c++11")

add_subdirectory(src)
