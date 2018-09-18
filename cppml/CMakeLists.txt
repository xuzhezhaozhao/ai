project(cppml)

cmake_minimum_required(VERSION 2.8)
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR "in source build is not allowed.")
endif()

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")

# gtest
include_directories(SYSTEM ../../deps/googletest/googletest/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../deps/googletest/build/googlemock/gtest/)

# jsoncpp
set(JSONCPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../deps/jsoncpp/)
include_directories(SYSTEM ${JSONCPP_DIR}/dist)

add_executable(jsoncpp_helper_test jsoncpp_helper_test.cpp ${JSONCPP_DIR}/dist/jsoncpp.cpp)
target_link_libraries(jsoncpp_helper_test gtest pthread)

# unittest
add_custom_target(unittest ./jsoncpp_helper_test DEPENDS jsoncpp_helper_test)
