# Allow overriding XTensor versions, e.g., for testing a new version in DP3. For
# avoiding ODR violations, repositories that use aocommon should not override
# these versions in their master branch. That way, the XTensor versions will be
# equal in all repositories.
if(NOT xtl_GIT_TAG)
  set(xtl_GIT_TAG d11fb6b5f4c417025124ed2c62175284846a1914)
endif()
if(NOT xsimd_GIT_TAG)
  set(xsimd_GIT_TAG 58cd982144a8799ba736c8e0bb035ecf6ecadcfc)
endif()
if(NOT xtensor_GIT_TAG)
  set(xtensor_GIT_TAG ae52796961d03e7a3d754d72713be5098ce467b9)
endif()
if(NOT xtensor-blas_GIT_TAG)
  set(xtensor-blas_GIT_TAG 0.20.0)
endif()
if(NOT xtensor-fftw_GIT_TAG)
  set(xtensor-fftw_GIT_TAG e6be85a376624da10629b6525c81759e02020308)
endif()
if(NOT xtensor-python_GIT_TAG)
  set(xtensor-python_GIT_TAG 0.27.0)
endif()

# By default, only load the basic 'xtensor' library.
if(NOT XTENSOR_LIBRARIES)
  set(XTENSOR_LIBRARIES xtensor)
endif()

# The 'xtensor' library requires the 'xtl' library.
if(NOT xtl IN_LIST XTENSOR_LIBRARIES)
  list(APPEND XTENSOR_LIBRARIES xtl)
endif()

include(FetchContent)

foreach(LIB ${XTENSOR_LIBRARIES})
  set(XT_GIT_TAG "${${LIB}_GIT_TAG}")
  if(NOT XT_GIT_TAG)
    message(FATAL_ERROR "Unknown git tag for XTensor library '${LIB}'")
  endif()

  # Checking out a specific git commit hash does not (always) work when
  # GIT_SHALLOW is TRUE. See the documentation for GIT_TAG in
  # https://cmake.org/cmake/help/latest/module/ExternalProject.html -> If the
  # GIT_TAG is a commit hash, use a non-shallow clone.
  string(LENGTH "${XT_GIT_TAG}" XT_TAG_LENGTH)
  set(XT_SHALLOW TRUE)
  if(XT_TAG_LENGTH EQUAL 40 AND XT_GIT_TAG MATCHES "^[0-9a-f]+$")
    set(XT_SHALLOW FALSE)
  endif()

  FetchContent_Declare(
    ${LIB}
    GIT_REPOSITORY https://github.com/xtensor-stack/${LIB}.git
    GIT_SHALLOW ${XT_SHALLOW}
    GIT_TAG ${XT_GIT_TAG})

  # Use FetchContent_MakeAvailable to automatically handle the library setup
  FetchContent_MakeAvailable(${LIB})
endforeach()

# Enable XSIMD if it's included in the libraries
if(xsimd IN_LIST XTENSOR_LIBRARIES)
  add_compile_definitions(XTENSOR_USE_XSIMD)
endif()

# This option is ON by default in xtensor, since avoiding temporaries in
# assignments is still experimental.
# https://github.com/xtensor-stack/xtensor/issues/2819 also shows an issue.
add_compile_definitions(XTENSOR_FORCE_TEMPORARY_MEMORY_IN_ASSIGNMENTS)
