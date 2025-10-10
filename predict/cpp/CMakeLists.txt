cmake_minimum_required(VERSION 3.14)
project(predict LANGUAGES CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CMAKE_BUILD_TYPE "" RelWithDebInfo)
option(PREDICT_BUILD_TESTING "Build the test suite" OFF)
option(PREDICT_BUILD_BENCHMARK "Build the benchmark suite" OFF)
option(PREDICT_DIAGNOSTICS "Enable diagnostics output" OFF)
option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compile commands" ON)
option(PREDICT_PORTABLE
       "Build portable binaries (with slightly decreased performance)" OFF)

# Find Boost components
find_package(Boost REQUIRED COMPONENTS unit_test_framework)

find_package(HDF5 REQUIRED COMPONENTS CXX)
add_definitions(${HDF5_DEFINITIONS})

# Investigate xtensor dependencies. Only include FetchXTensor if any of the
# targets are missing.
set(XTENSOR_LIBRARIES "")

if(NOT TARGET xtl)
  list(APPEND XTENSOR_LIBRARIES xtl)
endif()

if(NOT TARGET xsimd)
  list(APPEND XTENSOR_LIBRARIES xsimd)
endif()

if(NOT TARGET xtensor)
  list(APPEND XTENSOR_LIBRARIES xtensor)
endif()

if(NOT TARGET xtensor-blas)
  list(APPEND XTENSOR_LIBRARIES xtensor-blas)
endif()

# Only include FetchXTensor if any of the targets are missing
if(XTENSOR_LIBRARIES)
  include("${CMAKE_SOURCE_DIR}/cmake/FetchXTensor.cmake")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      RelWithDebInfo
      CACHE STRING "Choose the type of build." FORCE)
endif()

# Compiler warning configuration is based on DP3: https://git.astron.nl/RD/DP3
set(PREDICT_BUILD_ARGUMENTS_BASE
    -Wall
    -Wnon-virtual-dtor
    # -Wzero-as-null-pointer-constant
    -Wduplicated-branches
    # -Wundef
    -Wvla
    -Wpointer-arith
    -Wextra
    -Wno-unused-parameter
    $<$<BOOL:${PREDICT_DIAGNOSTICS}>:-DPREDICT_DIAGNOSTICS>)

include(cmake/SetTargetCPU.cmake)

# Find BLAS
find_package(BLAS REQUIRED)

find_package(OpenMP REQUIRED)

# TrigDx for fast trigonometric operations
FetchContent_Declare(
  trigdx
  GIT_REPOSITORY https://github.com/astron-rd/TrigDx.git
  GIT_TAG main
  GIT_SHALLOW true
  GIT_PROGRESS true)

set(TRIGDX_USE_XSIMD ON)
set(TRIGDX_BUILD_TESTS OFF)
set(TRIGDX_BUILD_BENCHMARKS OFF)
set(TRIGDX_BUILD_PYTHON OFF)

FetchContent_MakeAvailable(trigdx)

file(GLOB_RECURSE SOURCES src/*.cpp)
file(GLOB_RECURSE HEADERS include/*.h)

# Library target
add_library(predict STATIC ${SOURCES})

target_compile_options(predict PRIVATE ${PREDICT_BUILD_ARGUMENTS})
target_link_options(predict PRIVATE ${PREDICT_BUILD_ARGUMENTS})

target_include_directories(
  predict PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                 $<INSTALL_INTERFACE:include>)

target_link_libraries(
  predict
  PUBLIC xtensor
         xtensor-blas
         xsimd
         OpenMP::OpenMP_CXX
         trigdx
         ${BLAS_LIBRARIES}
         $<$<BOOL:${ENABLE_TRACY_PROFILING}>:Tracy::TracyClient>)

message(STATUS "Boost include dir: ${Boost_INCLUDE_DIRS}")

target_include_directories(predict PRIVATE ${trigdx_SOURCE_DIR}/include
                                           ${Boost_INCLUDE_DIRS})

# clang-tidy
find_program(
  CLANG_TIDY_EXE
  NAMES clang-tidy
  HINTS ~/.local/bin)

if(CLANG_TIDY_EXE)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
else()
  message("clang-tidy not found. Skipping compile_commands.json generation.")
endif()

# Tests
file(GLOB TESTS tests/test_*.cpp)

# Add benchmark subdirectory
if(PREDICT_BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()

if(PREDICT_BUILD_TESTING)
  # Enable testing
  include(CTest)
  enable_testing()

  # Find boost unittest framework
  find_package(Boost REQUIRED COMPONENTS unit_test_framework)
  file(GLOB TESTS tests/test_*.cpp)

  foreach(test ${TESTS})
    get_filename_component(test_name ${test} NAME_WE)
    string(REPLACE "test_" "" test_name ${test_name})
    message(STATUS "Adding Predict test: ${test_name}")
    add_test(NAME ${test_name}_unittests COMMAND ${test_name}_unittests)
    add_executable(${test_name}_unittests ${test})

    target_include_directories(
      ${test_name}_unittests
      PRIVATE ${CASACORE_INCLUDE_DIRS} ${aocommon_SOURCE_DIR}/include
              ${EVERYBEAM_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

    # Link Boost test framework to the test executable
    target_link_libraries(
      ${test_name}_unittests PRIVATE predict Boost::unit_test_framework xtensor
                                     xtensor-blas EveryBeam::everybeam)

    set_source_files_properties(${test} PROPERTIES COMPILE_DEFINITIONS
                                                   "BOOST_TEST_DYN_LINK")
    # Add a test that runs the test executable
    target_compile_options(${test_name}_unittests
                           PRIVATE -coverage ${PREDICT_BUILD_ARGUMENTS})
    target_link_options(${test_name}_unittests PRIVATE -coverage
                        $<$<CONFIG:Debug>:-g>)
  endforeach()
endif()

option(ENABLE_TRACY_PROFILING "Enables compilation with the Tracy profiler" OFF)

if(ENABLE_TRACY_PROFILING)
  option(ENABLE_TRACY "Enable tracy profiling" ON)
  FetchContent_Declare(
    tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_TAG v0.12.2 # Or pin to a specific version
    GIT_SHALLOW true
    GIT_PROGRESS true)
  FetchContent_MakeAvailable(tracy)
  target_compile_definitions(predict PRIVATE ENABLE_TRACY_PROFILING)
endif()
