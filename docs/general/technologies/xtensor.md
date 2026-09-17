# XTensor

## Overview

[*xtensor*](https://xtensor.readthedocs.io/en/latest/) is a C++ library for working with multi-dimensional arrays, inspired by [NumPy](https://numpy.org/). This involves the basic *N*-dimensional structures, but also an entire expression engine that allows numerical computations on any object that implements the expression interface, such as the containers; these expressions are lazy.

The library also comes with *adapters* to convert, say, C-style arrays or standard library vectors, into their expression system.

Moreover, they provide a lot of convenient functions, such as initializers, slicing, concatenation, rearranging elements, reducers, NaN functions, and much more. For readers familiar with NumPy, the following page ["From NumPy to xtensor"](https://xtensor.readthedocs.io/en/latest/numpy.html), describes a lot of these functions and show the NumPy counterparts.

### The whole family

*xtensor* is but a small part of the whole [*xtensor* stack](https://github.com/xtensor-stack), which consits of of various libraries that extend *xtensor*, such as:

- *xsimd* for working with SIMD intrinsics,
- *xtensor-fftw* for performing FFTs using [FFTW](https://fftw.org/),
- *xtensor-blas* for linear algebra (similar to NumPy's `linalg`),
- *xtensor-io* for reading common file formats,
- *xtensor-python* for using NumPy data structures from C++

Of this family, we used `xtensor`, `xtensor-fftw`, and briefly investigated `xtensor-io` for reading binary `.npy` files.

## The good, the bad, and the ugly

*xtensor* offers an incredibly convenient way to work with multi-dimensional vectors, especially if you're introduction to scientific computing is with Python / NumPy. Defining *N*-dimensional arrays and manipulating them feels very natural, espcially, if you compare it to using C++'s standard library *std::vector*, and many operators that would require (multiple levels of) loops, are possible with a single function call. Thus, code that uses *xtensor* tends to be easier to read.

In particular, this is the case for *xtensor-fftw* that wraps the commonly used *FFTW3* library for computing the FFT; an operation that is omnipresent in radio astronomy. FFTW3 is written in portable C, thus offers a (rather archaic) C-style interface, which requires a lot of manual memory management and can therefore be quite error prone. However, *xtensor-fftw* isn't as mature as *xtensor* itself and isn't very actively developed, although it seems they do keep it up-to-date. They, for example, lack some convenience functions that one might expect coming from NumPy, such as a multi-dimensional `fftshift` operation.

Lastly, we investigated the use of `xtensor-io` for storing and loading data stored in the `.npy` format accross all languages, since most languages offer a library with such functionality. Sadly, it does not support compound data types, which means having to store members of, e.g., a `struct` in different file.

## `xtensor-fftw` bechmark

The following table compares the performance of *xtensor-fftw* (left) vs. FFTW3 (right). It's clear that using FFTW3 natively is the more performant solution, especially for smaller volume FFTs, FFTW is a lot (~100) times faster. For larger data volumes, the difference is signficantly smaller, and in case of the complex-to-complex FFT, is as small as (roughly) a factor 4.

The huge difference can likely (at least) partially be attributed to the fact that every call to *xtensor*'s FFT routines, create a new FFTW plan, while FFTW itself supports reusing the plan.

Note that these values were obtained using the [`fft-benchmark`](https://git.astron.nl/RD/fft-benchmark) tool.

```

| Operation | xtensor-fftw time | FFTW3 time | Speedup (FFTW3 over xtensor-fftw) |
| :--- | :---: | :---: | :---: |
| R2C/100 | 11.6 us | 0.160 us | $\approx 72.5\times$ |
| R2C/200 | 23.1 us | 0.301 us | $\approx 76.8\times$ |
| R2C/300 | 20.7 us | 0.582 us | $\approx 35.6\times$ |
| R2C/400 | 21.1 us | 0.766 us | $\approx 27.6\times$ |
| R2C/500 | 22.2 us | 0.976 us | $\approx 22.7\times$ |
| R2C/600 | 22.6 us | 1.13 us | $\approx 20.0\times$ |
| R2C/700 | 34.8 us | 1.87 us | $\approx 18.6\times$ |
| R2C/800 | 25.1 us | 1.53 us | $\approx 16.4\times$ |
| R2C/900 | 34.2 us | 2.64 us | $\approx 12.9\times$ |
| R2C/1000 | 35.4 us | 1.96 us | $\approx 18.1\times$ |
| C2R/100 | 12.5 us | 0.162 us | $\approx 77.2\times$ |
| C2R/200 | 23.7 us | 0.302 us | $\approx 78.5\times$ |
| C2R/300 | 22.2 us | 0.615 us | $\approx 36.1\times$ |
| C2R/400 | 22.9 us | 0.814 us | $\approx 28.1\times$ |
| C2R/500 | 23.8 us | 1.05 us | $\approx 22.7\times$ |
| C2R/600 | 24.6 us | 1.23 us | $\approx 20.0\times$ |
| C2R/700 | 35.6 us | 1.88 us | $\approx 18.9\times$ |
| C2R/800 | 26.4 us | 1.63 us | $\approx 16.2\times$ |
| C2R/900 | 28.4 us | 1.80 us | $\approx 15.8\times$ |
| C2R/1000 | 38.0 us | 2.13 us | $\approx 17.8\times$ |
| C2C/100 | 6.64 us | 0.210 us | $\approx 31.6\times$ |
| C2C/200 | 7.58 us | 0.433 us | $\approx 17.5\times$ |
| C2C/300 | 15.3 us | 1.14 us | $\approx 13.4\times$ |
| C2C/400 | 9.03 us | 0.895 us | $\approx 10.1\times$ |
| C2C/500 | 17.2 us | 1.85 us | $\approx 9.3\times$ |
| C2C/600 | 17.2 us | 2.38 us | $\approx 7.2\times$ |
| C2C/700 | 17.5 us | 2.82 us | $\approx 6.2\times$ |
| C2C/800 | 11.1 us | 2.48 us | $\approx 4.5\times$ |
| C2C/900 | 28.6 us | 5.33 us | $\approx 5.4\times$ |
| C2C/1000 | 20.0 us | 4.35 us | $\approx 4.6\times$ |
```
