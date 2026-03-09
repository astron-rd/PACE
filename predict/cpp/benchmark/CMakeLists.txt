include(FetchContent)

# Google benchmarrk
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.2)

set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)

# Make the fetched content available.
FetchContent_MakeAvailable(googlebenchmark)

set(BENCHMARKS simulator)

add_executable(
  microbenchmarks
  main.cpp
  predict.cpp
  directions.cpp
  spectrum.cpp
  phases.cpp
  spectrum_cross.cpp
  smearterms.cpp)

target_compile_options(microbenchmarks PRIVATE ${PREDICT_BUILD_ARGUMENTS})

target_link_libraries(
  microbenchmarks
  PRIVATE benchmark::benchmark
          predict
          xtensor
          xtensor-blas
          xsimd
          ${CASACORE_LIBRARIES}
          $<$<BOOL:${ENABLE_TRACY_PROFILING}>:Tracy::TracyClient>)

target_include_directories(
  microbenchmarks
  PRIVATE include
          ${CMAKE_SOURCE_DIR}/include
          ${CASACORE_INCLUDE_DIRS}
          ${tracy_SOURCE_DIR}/include
          ${aocommon_SOURCE_DIR}/include
          ${Boost_INCLUDE_DIRS}
          ${EVERYBEAM_INCLUDE_DIRS})
