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

option(DEDISP_DEBUG_HDF5 "Dump intermediate XTensor arrays to an HDF5 file" OFF)
if(DEDISP_DEBUG_HDF5)
  add_definitions(-DDEDISP_DEBUG_HDF5)
endif()

option(DEDISP_USE_OPENMP "Use OpenMP to parallise loops" ON)
if(DEDISP_USE_OPENMP)
  add_definitions(-DDEDISP_USE_OPENMP)
endif()

option(DEDISP_BENCHMARK "Print dedispersion timings" ON)
if(DEDISP_BENCHMARK)
  add_definitions(-DDEDISP_BENCHMARK)
endif()

# Load external dependencies
include(FetchContent)

set(XTENSOR_LIBRARIES xtl xtensor xtensor-fftw-float)
include("${CMAKE_SOURCE_DIR}/cmake/FetchXTensor.cmake")
FetchContent_MakeAvailable(xtensor)

include("${CMAKE_SOURCE_DIR}/cmake/FetchH5cpp.cmake")

find_package(OpenMP)

# Configure targets
add_library(dedisp OBJECT src/fddplan.cpp src/kernels.cpp)
target_link_libraries(dedisp PUBLIC xtensor xtensor-fftw OpenMP::OpenMP_CXX
                                    h5cpp::h5cpp)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE dedisp h5cpp::h5cpp)
target_include_directories(main PRIVATE ${PROJECT_SOURCE_DIR}/include)
