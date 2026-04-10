cmake_minimum_required(VERSION 3.14)
project(RadioAstronomy CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# To use clangd as LSP
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}")

# Add various debugging options
option(DEDISP_DEBUG "Increase logging verbosity for debugging" OFF)
if(DEDISP_DEBUG)
  add_definitions(-DDEDISP_DEBUG)
endif()

option(DEDISP_DEBUG_NPY "Dump intermediate XTensor arrays as NPY files" OFF)
if(DEDISP_DEBUG_NPY)
  add_definitions(-DDEDISP_DEBUG_NPY)
endif()

option(DEDISP_USE_OPENMP "Use OpenMP to parallise loops" ON)
if(DEDISP_USE_OPENMP)
  add_definitions(-DDEDISP_USE_OPENMP)
endif()

# Load external dependencies
include(FetchContent)

set(XTENSOR_LIBRARIES xtl xtensor xtensor-fftw-float)
include("${CMAKE_SOURCE_DIR}/cmake/FetchXTensor.cmake")

FetchContent_MakeAvailable(xtensor)

find_package(OpenMP)

# Configure targets
add_library(dedisp OBJECT src/fddplan.cpp src/kernels.cpp src/utilities.cpp)
target_link_libraries(dedisp PUBLIC xtensor xtensor-fftw OpenMP::OpenMP_CXX)

add_executable(testfdd bin/testfdd.cpp)
target_link_libraries(testfdd PRIVATE dedisp)
target_include_directories(testfdd PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(testfil bin/testfil.cpp)
target_link_libraries(testfil PRIVATE dedisp)
target_include_directories(testfil PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(testroutines bin/testroutines.cpp)
target_link_libraries(testroutines PRIVATE dedisp)
target_include_directories(testroutines PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(simulatesignal bin/simulatesignal.cpp)
target_link_libraries(simulatesignal PRIVATE dedisp)
