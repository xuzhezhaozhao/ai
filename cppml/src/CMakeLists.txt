
# gtest
include_directories(SYSTEM ../../deps/googletest/googletest/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../deps/googletest/build/googlemock/gtest/)

add_executable(transformer_test transformer_test.cpp)
target_link_libraries(transformer_test gtest pthread)

# unittest
add_custom_target(unittest ./transformer_test DEPENDS transformer_test)
