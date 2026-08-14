# FetchXTensor.cmake Future-proof helper using ExternalProject_Add
#
# Fetches xtensor-stack components without polluting installation.
#
# cmake-lint: disable=C0103

cmake_minimum_required(VERSION 3.21)

include(ExternalProject)

# ---------------------------------------------------------------------------
# Default tags
# ---------------------------------------------------------------------------
if(NOT xtl_GIT_TAG)
  set(xtl_GIT_TAG 0.8.0)
endif()
if(NOT xsimd_GIT_TAG)
  set(xsimd_GIT_TAG 13.2.0)
endif()
if(NOT xtensor_GIT_TAG)
  set(xtensor_GIT_TAG 0.27.0)
endif()
if(NOT xtensor-blas_GIT_TAG)
  set(xtensor-blas_GIT_TAG 0.20.0)
endif()
if(NOT xtensor-fftw_GIT_TAG)
  set(xtensor-fftw_GIT_TAG 127aae73d9def7d421045d1fe5cf6b9c73da3542)
endif()
if(NOT xtensor-python_GIT_TAG)
  set(xtensor-python_GIT_TAG 0.28.0)
endif()
if(NOT xtensor-io_GIT_TAG)
  set(xtensor-io_GIT_TAG 0.3.0)
endif()

# ---------------------------------------------------------------------------
# Library list
# ---------------------------------------------------------------------------
if(NOT XTENSOR_LIBRARIES)
  set(XTENSOR_LIBRARIES xtensor)
endif()

# xtensor always needs xtl
if(NOT xtl IN_LIST XTENSOR_LIBRARIES)
  list(APPEND XTENSOR_LIBRARIES xtl)
endif()

# ---------------------------------------------------------------------------
# Function: fetch one component using ExternalProject_Add
# ---------------------------------------------------------------------------
function(xt_fetch_source_only LIB GIT_TAG)
  string(LENGTH "${GIT_TAG}" tag_len)
  set(shallow TRUE)
  if(tag_len EQUAL 40 AND GIT_TAG MATCHES "^[0-9a-f]+$")
    set(shallow FALSE)
  endif()

  set(src_dir "${CMAKE_BINARY_DIR}/_deps/${LIB}-src")

  ExternalProject_Add(
    ${LIB}_ext
    PREFIX "${CMAKE_BINARY_DIR}/_deps/${LIB}"
    GIT_REPOSITORY "https://github.com/xtensor-stack/${LIB}.git"
    GIT_TAG "${GIT_TAG}"
    GIT_SHALLOW ${shallow}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_DISCONNECTED TRUE
    SOURCE_DIR "${src_dir}")

  # INTERFACE target
  if(NOT TARGET ${LIB})
    add_library(${LIB} INTERFACE)
    add_dependencies(${LIB} ${LIB}_ext)
    target_include_directories(${LIB} SYSTEM INTERFACE "${src_dir}/include")
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Fetch all requested libraries
# ---------------------------------------------------------------------------
foreach(LIB ${XTENSOR_LIBRARIES})
  set(XT_GIT_TAG "${${LIB}_GIT_TAG}")
  if(NOT XT_GIT_TAG)
    message(FATAL_ERROR "Unknown git tag for XTensor library '${LIB}'")
  endif()
  xt_fetch_source_only(${LIB} "${XT_GIT_TAG}")
endforeach()

# ---------------------------------------------------------------------------
# Dependencies and compile definitions
# ---------------------------------------------------------------------------
target_link_libraries(xtensor INTERFACE xtl)

if(xsimd IN_LIST XTENSOR_LIBRARIES)
  target_link_libraries(xtensor INTERFACE xsimd)
  target_compile_definitions(xtensor INTERFACE XTENSOR_USE_XSIMD)
endif()

target_compile_definitions(
  xtensor INTERFACE XTENSOR_FORCE_TEMPORARY_MEMORY_IN_ASSIGNMENTS)
