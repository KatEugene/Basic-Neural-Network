cmake_minimum_required(VERSION 3.14)

project(BasicNeuralNetwork)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(libs/GoogleTest)

include(src/sources.cmake)
include(tests/sources.cmake)

enable_testing()

target_link_libraries(
  test_neural_network 
  neural_network
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_neural_network)
