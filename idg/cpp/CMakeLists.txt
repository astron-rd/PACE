cmake_minimum_required(VERSION 3.14)
project(RadioAstronomy CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3f IMPORTED_TARGET)

include(FetchContent)

set(XTENSOR_LIBRARIES xtl xtensor xtensor-fftw xtensor-io)
include("${CMAKE_SOURCE_DIR}/cmake/FetchXTensor.cmake")

find_package(OpenMP)

add_subdirectory(src)
