cmake_minimum_required(VERSION 3.14)

project(BasicNeuralNetwork)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://github.com/ValveSoftware/eigen.git
  GIT_TAG 3147391d946bb4b6c68edd901f2add6ac1f31f8c
)
FetchContent_Declare(
  EigenRand
  GIT_REPOSITORY https://github.com/bab2min/EigenRand.git
  GIT_TAG 13582eb26e6f7f67502c1732f13ad37498152be1
)

FetchContent_MakeAvailable(googletest)
FetchContent_MakeAvailable(eigen)
FetchContent_MakeAvailable(EigenRand)

include(src/sources.cmake)
include(tests/sources.cmake)

enable_testing()

target_link_libraries(
  test_neural_network 
  neural_network
  GTest::gtest_main
  eigen
  EigenRand
)

include(GoogleTest)
gtest_discover_tests(test_neural_network)
