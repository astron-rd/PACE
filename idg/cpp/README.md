# C++ IDG implementation

This is a C++ implementation of IDG. It loads visibilities from an input file and uses Image Domain Gridding to create an image.

For more information on the input data format, see the [Data Format documentation](../../docs/general/data-format.md).

## Building

To build this application you will need a recent C++ compiler and CMake.

1. Install the HDF5 and FFTW libraries on your system. You can use the system package manager or the module system if you're building on an HPC cluster. The libraries need to be exposed so `pkg-config` can find them.
1. Configure the application with `cmake -B build`. You can also customize the build interactively with `ccmake -B build`.
1. Build the application with `cmake --build build -j`. If the build fails or runs out of memory you can omit `-j` or specify a number of cores with `-j num-cores`.

Your executable will be `build/src/main[.exe]`.

## Usage

To load an input file and grid an image, just run `main`. This will load input from `input.h5` in the current working directory. This path can be customized with the `--input_path` flag.

By default, IDG will only output the finished image. To also output the subgrids, pass the `--output_subgrids` option.

## Parallelization

This implementation uses OpenMP to parallelize the gridding kernels. The primary parallelization is achieved by distributing the processing of subgrids across multiple CPU cores.

Specifically, the `grid_onto_subgrids` function in `idg/cpp/src/IDG.cpp` uses the `#pragma omp parallel for` directive with a `dynamic` schedule:

```cpp
#pragma omp parallel for schedule(dynamic)
for (size_t s = 0; s < nr_subgrids; ++s) {
    // Process subgrids
}
```

- **`#pragma omp parallel for`**: This tells the compiler to parallelize the following `for` loop, splitting the iterations among the available threads in the OpenMP pool.
- **`schedule(dynamic)`**: Since the amount of work per subgrid can vary depending on the number of visibilities assigned to it, a dynamic schedule is used to balance the load. Threads are assigned iterations dynamically as they become available, preventing some cores from idling while others are still working on "heavy" subgrids.

Other components of the pipeline could potentially be parallelized, but were omitted for simplicity in this reference implementation:

- **`transform`**: Could be parallelized over the 4 polarizations, or more extensively by breaking down the FFT into rows and columns.
- **`add_subgrids_to_grid`**: Parallelization is more complex here because subgrids may partially overlap, requiring careful synchronization or atomic operations to avoid race conditions.

Since `grid_onto_subgrids` is the main computational bottleneck, it is the only part of this reference implementation that utilizes OpenMP.

You can control the number of threads used by setting the `OMP_NUM_THREADS` environment variable.
