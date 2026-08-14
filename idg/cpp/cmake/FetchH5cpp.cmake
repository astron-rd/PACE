set(H5CPP_CONAN
    DISABLE
    CACHE STRING "")
set(H5CPP_DISABLE_TESTS
    ON
    CACHE STRING "")
FetchContent_Declare(
  h5cpp
  GIT_REPOSITORY https://github.com/ess-dmsc/h5cpp.git
  GIT_TAG v0.8.0)
FetchContent_MakeAvailable(h5cpp)
