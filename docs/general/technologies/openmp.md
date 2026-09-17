# OpenMP

## Overview

OpenMP is a multiprocessing library, which provides easy to use directives to tell the compiler how the code should be executed in parallel. In particular, it provides an API for writing shared memory parallel applications in C, C++, and Fortran.

## Functionality

As aluded to in the overview, OpenMP, offers a wide range of directives for running parts of a C++ program in parallel. For instance, the following directive executes a block in a different thread:

```cpp
#pragma omp parallel
{
    // expensive processing
}
```

In the dedispersion application, we perform a lot of operations for all frequency channels in a dynamic spectrum, however, all of this channel data is independent, thus parallising over a loop over channels provides a large speed-up and is as simple as adding a single directive:

```cpp
#pragma omp parallel for
for (size_t d = 0; d < dm_count_; ++d) {
  xt::xarray<std::complex<float>> samples = xt::eval(xt::row(dm_scratch, d));
  xt::view(dm_data, d, xt::all()) = xt::fftw::irfft(samples);
}
```

Besides parallelisation over multiple cores, OpenMP also supports [offloading to accelerators since version 4](https://www.openmp.org/updates/openmp-accelerator-support-gpus/). Offloading requires a simple directive: [`#pragma omp target`](https://www.openmp.org/spec-html/5.2/openmpse85.html), the target device can be configured through the [device clause](https://www.openmp.org/spec-html/5.2/openmpse79.html); exploration of this functionality will be part of a future deliverable. This directive instructs the compiler to generate a so-called target task, which maps variables to a device data environment and executes the enclosed block of code on that device.
