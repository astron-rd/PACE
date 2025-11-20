# Allow overriding XTensor versions, e.g., for testing a new version in DP3. For
# avoiding ODR violations, repositories that use aocommon should not override
# these versions in their master branch. That way, the XTensor versions will be
# equal in all repositories.

# cmake-lint: disable=C0103
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
  if(LIB MATCHES "fftw")
    string(REGEX REPLACE "^(.*)-([^-]*)$" "\\1;\\2" LIB_PARTS ${LIB})
    message("${LIB}")
    message("${LIB_PARTS}")
    list(GET LIB_PARTS 0 LIB_BASE)
    list(GET LIB_PARTS 1 FFTW_PRECISION)

    string(TOUPPER ${FFTW_PRECISION} FFTW_PRECISION)
    set(HAVE_XTENSOR_FFTW_${FFTW_PRECISION} TRUE)
    set(XT_GIT_TAG ${${LIB_BASE}_GIT_TAG})

    set(LIB ${LIB_BASE})
  else()
    set(XT_GIT_TAG "${${LIB}_GIT_TAG}")
  endif()
  if(NOT XT_GIT_TAG)
    message(FATAL_ERROR "Unknown git tag for XTensor library ${LIB}")
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

  if(NOT TARGET ${LIB})
    FetchContent_Declare(
      ${LIB}
      GIT_REPOSITORY https://github.com/xtensor-stack/${LIB}.git
      GIT_SHALLOW ${XT_SHALLOW}
      GIT_TAG ${XT_GIT_TAG})

    # FetchContent_MakeAvailable makes ${LIB} part of the project. Headers from
    # ${LIB} are then installed along with the project. However, most projects
    # only use ${LIB} internally, at compile time, and should not install ${LIB},
    # including its headers: - For libraries, XTensor shouldn't be part the public
    # API. - For applications, installing headers isn't needed at all.
    #
    # Instead of FetchContent_MakeAvailable, we therefore use
    # FetchContent_Populate and define an INTERFACE target manually. This approach
    # also supports xtensor-fftw, which does not define a CMake target and also
    # loads FFTW using custom options A drawback of this approach is that we have
    # to set some options manually, like
    # XTENSOR_FORCE_TEMPORARY_MEMORY_IN_ASSIGNMENTS.

    FetchContent_GetProperties(${LIB})
    if(NOT ${${LIB}_POPULATED})
      FetchContent_Populate(${LIB})
    endif()

    add_library(${LIB} INTERFACE)
    target_include_directories(${LIB} SYSTEM
                              INTERFACE "${${LIB}_SOURCE_DIR}/include")
  endif()
endforeach()

# Since xtensor uses xtl and possibly xsimd headers, create dependencies.
target_link_libraries(xtensor INTERFACE xtl)
if(xsimd IN_LIST XTENSOR_LIBRARIES)
  target_link_libraries(xtensor INTERFACE xsimd)
  add_compile_definitions(XTENSOR_USE_XSIMD)
endif()

# Since xtensor-fftw uses fftw3, link fftw3(f) library if it's found.
if(HAVE_XTENSOR_FFTW_DOUBLE OR HAVE_XTENSOR_FFTW_FLOAT)
  find_package(PkgConfig)
  if(NOT PkgConfig_FOUND)
    message(WARNING "PkgConfig not found, not linking xtensor-fftw to fftw3.")
  else()
    pkg_search_module(FFTW fftw3 IMPORTED_TARGET)
    if(FFTW_FOUND AND HAVE_XTENSOR_FFTW_DOUBLE)
      target_link_libraries(xtensor-fftw INTERFACE PkgConfig::FFTW)
    endif()

    pkg_search_module(FFTWF fftw3f IMPORTED_TARGET)
    if(FFTWF_FOUND AND HAVE_XTENSOR_FFTW_FLOAT)
      target_link_libraries(xtensor-fftw INTERFACE PkgConfig::FFTWF)
    endif()

    if(NOT FFTW_FOUND AND NOT FFTWF_FOUND)
      message(WARNING "Can't find fftw3 nor fftw3f.")
    endif()
  endif()
endif()

# This option is ON by default in xtensor, since avoiding temporaries in
# assignments is still experimental.
# https://github.com/xtensor-stack/xtensor/issues/2819 also shows an issue.
add_compile_definitions(XTENSOR_FORCE_TEMPORARY_MEMORY_IN_ASSIGNMENTS)
