project(fasttext)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -funroll-loops -pthread")

include_directories(./deps/lshbox)
include_directories(./deps/cppjieba/include)
include_directories(./deps/cppjieba/deps/)

add_subdirectory(src)
add_subdirectory(tests)
