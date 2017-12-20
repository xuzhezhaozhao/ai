project(generate_synonyms)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-g -O2 -Wall -std=c++11")

include_directories(./deps/fastText/src)

add_subdirectory(deps)
add_subdirectory(src)
