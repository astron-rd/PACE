# DEDISP: C++ Reference

This directory contains the `dedisp` (C++) reference code, based on the Fourier
Domain Dedispersion (FDD) method developed by
[C.G. Bassa et al. (2021)](https://git.astron.nl/RD/dedisp/).

## Usage

To run the FDD algorithm run `fdd`, this will try to load a dynamic spectrum from `signal.h5` in the current working directory. The result of the dedispersion results are written to a file called `fdd.h5`.

Note that these path can be customised using the `--spectrum` (input) and `--file` (output) options. Please use `-h` or `--help` to see all CLI options.

## Build Instructions

Ensure that you have `OpenMP` installed, then proceed to build the project with
CMake, e.g.:

```bash
mkdir build
cd build
cmake ..
```

The default CPU build needs a system HDF5 install at configure time (the h5cpp
dependency). The GPU offload example in `examples/` has no such dependency and
builds standalone.

If you are building on Apple hardware and have installed `OpenMP` using `brew`,
you may need to set `OpenMP_ROOT`:

```bash
cmake -DOpenMP_ROOT=$(brew --prefix)/opt/libomp ..
```

## GPU Offloading

The `fourier_domain_dedisperse` kernel can be offloaded to a GPU with OpenMP
target tasks (`-DDEDISP_GPU_BACKEND=omp`) or OpenACC (`=acc`), selected at
configure time. The pragma lives in `src/kernels.cpp`. The CPU path (`=off`,
the default) is unchanged. The kernel host buffers are pinned with NVHPC's
`-gpu=mem:separate:pinnedalloc`, which avoids the pageable-copy penalty.
See the [GPU offloading documentation](../../docs/general/technologies/gpu-offloading.md)
for the measured results and design decisions behind these flags.

### NVIDIA (NVHPC)

Build with the NVHPC `nvc++` compiler and set the backend and GPU target:

```bash
cmake -DCMAKE_CXX_COMPILER=nvc++ \
      -DDEDISP_GPU_BACKEND=omp   \
      -DDEDISP_GPU_TARGET=cc89,cc120 .
make
```

- `-DDEDISP_GPU_TARGET` defaults to `cc89,cc120`, one multi-arch fatbin that
  runs natively on both the AD4000 and the B4000. Narrow to `cc120` or `cc89`
  only when the target machine is known, because a single-arch binary crashes
  on the other card.
- `OMP_TARGET_OFFLOAD=mandatory ./fdd` makes host fallback fail loudly, to
  confirm real offload.

### AMD (ROCm clang)

ROCm's bundled clang compiles OpenMP target offload for the AMD Radeon AI Pro
R9700.
CMake's NVHPC flags are not portable, so for AMD compile the standalone bench
directly:

```bash
clang++ -O3 -std=c++20 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
        --offload-arch=gfx1201 -DDEDISP_GPU_TARGET_OMP \
        -Iinclude src/kernels.cpp examples/offload_bench.cpp -o offload_bench
OMP_TARGET_OFFLOAD=mandatory ./offload_bench
```

The same source offloads correctly on both vendors and matches the CPU result
to float32 rounding. `--offload-arch=gfx1201` is the runtime-visible
architecture the R9700 reports. `gfx1200` or `native` can silently
produce an image with no usable device. On the async question: OpenMP
`target ... nowait` on NVHPC works when paired with an explicit `#pragma omp taskwait` after the batches (verified on 26.3 and 26.9). NVHPC's implicit
synchronization path is what fails: returning from main or hitting a later
synchronous `target` region without an intervening taskwait hangs on 26.3 and
segfaults on 26.9. So use nowait with a taskwait, and fall back to OpenACC
queues only if the simpler full overlap is needed on NVIDIA.

Verify an offload build quickly on a GPU node:

```bash
nvc++ -O3 -std=c++20 -mp=gpu -gpu=cc120,mem:separate:pinnedalloc \
      -DDEDISP_GPU_TARGET_OMP -Iinclude \
      src/kernels.cpp examples/offload_bench.cpp -o offload_bench
./offload_bench
```
