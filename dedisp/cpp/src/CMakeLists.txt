include_directories("${CMAKE_SOURCE_DIR}/include")

add_library(dedisp OBJECT fdd.cpp)
target_link_libraries(dedisp PUBLIC xtensor xtensor-fftw PkgConfig::FFTW
                                 OpenMP::OpenMP_CXX)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE dedisp)