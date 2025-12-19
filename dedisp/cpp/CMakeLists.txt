cmake_minimum_required(VERSION 3.14)
project(RadioAstronomy CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}")

# Load external dependencies
include(FetchContent)

set(XTENSOR_LIBRARIES xtl xtensor xtensor-fftw-double xtensor-fftw-float)
include("${CMAKE_SOURCE_DIR}/cmake/FetchXTensor.cmake")

FetchContent_MakeAvailable(xtensor)

find_package(OpenMP)

# Configure targets
add_library(dedisp OBJECT src/fddplan.cpp src/kernels.cpp src/utilities.cpp)
target_link_libraries(dedisp PUBLIC xtensor xtensor-fftw OpenMP::OpenMP_CXX)

# add_executable(fdd src/main.cpp)
# target_link_libraries(fdd PRIVATE dedisp)

# install(TARGETS fdd DESTINATION bin)

add_executable(testfdd test/testfdd.cpp)
target_link_libraries(testfdd PRIVATE dedisp)
target_include_directories(testfdd PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(simulatesignal bin/simulatesignal.cpp)
target_link_libraries(simulatesignal PRIVATE dedisp)
