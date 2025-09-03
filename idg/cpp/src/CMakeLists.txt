include_directories("${CMAKE_SOURCE_DIR}/include")

add_library(init OBJECT init.cpp)
target_link_libraries(init PRIVATE xtensor)

add_library(kernels OBJECT kernels.cpp)
target_link_libraries(kernels PRIVATE xtensor)

add_library(idg OBJECT IDG.cpp $<TARGET_OBJECTS:init> $<TARGET_OBJECTS:kernels>)
target_link_libraries(idg PUBLIC xtensor xtensor-fftw PkgConfig::FFTW)
