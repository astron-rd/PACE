cmake_minimum_required(VERSION 3.14)
project(RadioAstronomy CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("${CMAKE_SOURCE_DIR}/include")

# Load external dependencies
include(FetchContent)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 fftw3f IMPORTED_TARGET)

set(XTENSOR_LIBRARIES xtl xtensor xtensor-fftw)
include("${CMAKE_SOURCE_DIR}/cmake/FetchXTensor.cmake")

FetchContent_MakeAvailable(xtensor)

find_package(OpenMP)

# Configure targets
add_library(dedisp OBJECT src/fddplan.cpp src/kernels.cpp)
target_link_libraries(dedisp PUBLIC xtensor xtensor-fftw PkgConfig::FFTW OpenMP::OpenMP_CXX)

add_executable(fdd src/main.cpp)
target_link_libraries(fdd PRIVATE dedisp)

install(TARGETS fdd DESTINATION bin)

add_executable(testfdd test/testfdd.cpp)
target_link_libraries(testfdd PRIVATE dedisp)
