# DEDISP: C++ Reference

This directory contains the `dedisp` (C++) reference code, based on the Fourier
Domain Dedispersion (FDD) method developed by
[C.G. Bassa et al. (2021)](https://git.astron.nl/RD/dedisp/).

## Usage

To run the FDD algorithm run `fdd`, this will try to load a dynamic spectrum from `signal.h5` in the current working directory. The result of the dedispersion results are written to a file called `fdd.h5`.

Note that these path can be customised using the `--spectrum` (input) and `--file` (output) options; pleaase use `-h` or `--help` to see all CLI options.

## Build Instructions

Ensure that you have `OpenMP` installed, then proceed to build the project with
CMake, e.g.:

```bash
mkdir build
cd build
cmake ..
```

If you are building on Apple hardware and have intalled `OpenMP` using `brew`,
you may need to set `OpenMP_ROOT`:

```bash
cmake -DOpenMP_ROOT=$(brew --prefix)/opt/libomp ..
```
