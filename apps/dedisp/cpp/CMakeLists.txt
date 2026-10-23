cmake_minimum_required(VERSION 3.21)
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

# GPU offloading backend for the dedispersion kernel: off | omp | acc. omp:
# OpenMP target offloading; acc: OpenACC. Build with the matching backend
# compiler (e.g. NVHPC nvc++ -mp=gpu or -acc); the pragma is in src/kernels.cpp,
# the flags below only add the compile/link switches. Use
# -DDEDISP_GPU_BACKEND=omp|acc together with -DCMAKE_CXX_COMPILER=nvc++ on the
# command line. The kernel host buffers are pinned via the
# -gpu=mem:separate:pinnedalloc route, which avoids pageable copies.
set(DEDISP_GPU_BACKEND
    "off"
    CACHE STRING
          "GPU offload backend for the dedispersion kernel: off, omp, acc")
set_property(CACHE DEDISP_GPU_BACKEND PROPERTY STRINGS off omp acc)
if(DEDISP_GPU_BACKEND STREQUAL "omp")
  add_definitions(-DDEDISP_GPU_TARGET_OMP)
elseif(DEDISP_GPU_BACKEND STREQUAL "acc")
  add_definitions(-DDEDISP_GPU_TARGET_ACC)
endif()

# Compute capability for the -gpu= flag. Default covers both gpu02 GPU families
# in one multi-arch fatbin: cc89 (Ada, RTX 4000) and cc120 (Blackwell, RTX PRO
# 4000). A single-arch build (cc120 only) crashes on the Ada cards, so narrow it
# only when the target machine is known.
set(DEDISP_GPU_TARGET
    "cc89,cc120"
    CACHE STRING
          "NVHPC -gpu= compute capability list, e.g. cc89,cc120 or cc120")

option(DEDISP_BENCHMARK "Print dedispersion timings" ON)
if(DEDISP_BENCHMARK)
  add_definitions(-DDEDISP_BENCHMARK)
endif()

# Load external dependencies
include(FetchContent)

FetchContent_Declare(
  xtensor-wrappers
  GIT_REPOSITORY https://github.com/astron-rd/xtensor-wrappers
  GIT_TAG main
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(xtensor-wrappers)

include("${CMAKE_SOURCE_DIR}/cmake/FetchH5cpp.cmake")

include("${CMAKE_SOURCE_DIR}/cmake/FetchCxxopts.cmake")

find_package(OpenMP)

# Configure targets
if(DEDISP_GPU_TARGET STREQUAL "" OR NOT DEDISP_GPU_TARGET MATCHES
                                    "^cc[0-9]+(,cc[0-9]+)*$")
  message(
    FATAL_ERROR "DEDISP_GPU_TARGET must be of the form ccNN or a comma list"
                "(e.g. cc120, cc89, cc89,cc120)")
endif()
set(DEDISP_GPU_FLAGS "-gpu=${DEDISP_GPU_TARGET},mem:separate:pinnedalloc")
if(DEDISP_GPU_BACKEND STREQUAL "acc")
  add_compile_options(-acc ${DEDISP_GPU_FLAGS})
  add_link_options(-acc ${DEDISP_GPU_FLAGS})
elseif(DEDISP_GPU_BACKEND STREQUAL "omp")
  add_compile_options(-mp=gpu ${DEDISP_GPU_FLAGS})
  add_link_options(-mp=gpu ${DEDISP_GPU_FLAGS})
endif()

add_library(dedisp OBJECT src/fddplan.cpp src/kernels.cpp)
target_link_libraries(dedisp PUBLIC cxxopts xtensor xtensor-wrappers
                                    OpenMP::OpenMP_CXX h5cpp::h5cpp)

add_executable(fdd src/main.cpp)
target_link_libraries(fdd PRIVATE dedisp h5cpp::h5cpp)
target_include_directories(fdd PRIVATE ${PROJECT_SOURCE_DIR}/include)
