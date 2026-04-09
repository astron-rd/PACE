# DEDISP: C++ Reference

This directory contains the `dedisp` (C++) reference code, based on the Fourier
Domain Dedispersion method developed by
[C.G. Bassa et al. (2021)](https://git.astron.nl/RD/dedisp/).

## Build Instructions

Ensure that you have `OpenMP` installed, then proceed to build the project with CMake, e.g.:
```bash
mkdir build
cd build
cmake ..
```

If you are building on Apple hardware and have intalled `OpenMP` using `brew`, you may need to set `OpenMP_ROOT`:
```bash
cmake -DOpenMP_ROOT=$(brew --prefix)/opt/libomp ..
```
