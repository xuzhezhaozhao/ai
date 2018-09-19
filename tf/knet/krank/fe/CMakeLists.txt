project(cppml)

cmake_minimum_required(VERSION 2.8)
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR "in source build is not allowed.")
endif()

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")

add_executable(preprocess preprocess.cpp stringpiece.cpp str_util.cpp)
