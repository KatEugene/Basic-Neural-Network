cmake_minimum_required(VERSION 3.13)

project(BasicNeuralNetwork)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG 344117638c8ff7e239044fd0fa7085839fc03021
)
FetchContent_Declare(
  xsimd
  GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
  GIT_TAG c1247bffa8fc36de7380a5cd42673a3b32f74c97
)

FetchContent_MakeAvailable(googletest)
FetchContent_MakeAvailable(benchmark)
FetchContent_MakeAvailable(xsimd)

include(src/sources.cmake)
include(tests/sources.cmake)

enable_testing()

target_link_libraries(test_neural_network neural_network GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_neural_network)