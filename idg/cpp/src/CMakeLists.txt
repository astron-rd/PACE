include_directories("${CMAKE_SOURCE_DIR}/include")

add_library(idg OBJECT IDG.cpp init.cpp kernels.cpp util.cpp)
target_link_libraries(idg PUBLIC cxxopts xtensor xtensor-fftw PkgConfig::FFTW
                                 OpenMP::OpenMP_CXX h5cpp::h5cpp)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE idg xtensor-io)
