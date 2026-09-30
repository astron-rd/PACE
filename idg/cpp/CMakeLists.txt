cmake_minimum_required(VERSION 3.21)
project(RadioAstronomy CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
  xtensor-wrappers
  GIT_REPOSITORY https://github.com/astron-rd/xtensor-wrappers
  GIT_TAG main
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(xtensor-wrappers)

include("${CMAKE_SOURCE_DIR}/cmake/FetchCxxopts.cmake")

include("${CMAKE_SOURCE_DIR}/cmake/FetchH5cpp.cmake")

find_package(OpenMP)

add_subdirectory(src)
