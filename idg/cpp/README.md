# C++ IDG implementation

This is a C++ implementation of IDG. It loads visibilities from an input file and uses Image Domain Gridding to create an image.

## Building

To build this application you will need a recent C++ compiler and CMake.

1. Install the HDF5 and FFTW libraries on your system. You can use the system package manager or the module system if you're building on an HPC cluster. The libraries need to be exposed so `pkg-config` can find them.
2. Configure the application with `cmake -B build`. You can also customize the build interactively with `ccmake -B build`.
3. Build the application with `cmake --build build -j`. If the build fails or runs out of memory you can omit `-j` or specify a number of cores with `-j num-cores`.

Your executable will be `build/src/main[.exe]`.

## Usage

To load an input file and grid an image, just run `main`. This will load input from `input.h5` in the current working directory. This path can be customized with the `--input_path` flag.

By default, IDG will only output the finished image. To also output the subgrids, pass the `--output_subgrids` option.
