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
-------------------------------------------------------------------------------------------------------------------------------
xtensor-fftw Benchmark        Time             CPU   Iterations        FFTW3 Benchmark        Time             CPU   Iterations
-------------------------------------------------------------------------------------------------------------------------------
XTENSORFFTW_R2C/100        11.6 us         11.6 us        60456        FFTW_R2C/100       0.160 us        0.160 us      4305547
XTENSORFFTW_R2C/200        23.1 us         23.1 us        30310        FFTW_R2C/200       0.301 us        0.300 us      2330772
XTENSORFFTW_R2C/300        20.7 us         20.7 us        33864        FFTW_R2C/300       0.582 us        0.582 us      1200359
XTENSORFFTW_R2C/400        21.1 us         21.1 us        33124        FFTW_R2C/400       0.766 us        0.766 us       912576
XTENSORFFTW_R2C/500        22.2 us         22.2 us        31440        FFTW_R2C/500       0.976 us        0.976 us       717027
XTENSORFFTW_R2C/600        22.6 us         22.6 us        30992        FFTW_R2C/600        1.13 us         1.13 us       616394
XTENSORFFTW_R2C/700        34.8 us         34.7 us        20201        FFTW_R2C/700        1.87 us         1.87 us       375502
XTENSORFFTW_R2C/800        25.1 us         25.1 us        27872        FFTW_R2C/800        1.53 us         1.52 us       462486
XTENSORFFTW_R2C/900        34.2 us         34.2 us        20443        FFTW_R2C/900        2.64 us         2.64 us       270997
XTENSORFFTW_R2C/1000       35.4 us         35.4 us        19743        FFTW_R2C/1000       1.96 us         1.96 us       355304
XTENSORFFTW_C2R/100        12.5 us         12.5 us        55970        FFTW_C2R/100       0.162 us        0.162 us      4413832
XTENSORFFTW_C2R/200        23.7 us         23.7 us        29496        FFTW_C2R/200       0.302 us        0.302 us      2308940
XTENSORFFTW_C2R/300        22.2 us         22.2 us        31540        FFTW_C2R/300       0.615 us        0.614 us      1136985
XTENSORFFTW_C2R/400        22.9 us         22.9 us        30551        FFTW_C2R/400       0.814 us        0.814 us       815105
XTENSORFFTW_C2R/500        23.8 us         23.8 us        29379        FFTW_C2R/500        1.05 us         1.05 us       664601
XTENSORFFTW_C2R/600        24.6 us         24.6 us        28433        FFTW_C2R/600        1.23 us         1.23 us       569011
XTENSORFFTW_C2R/700        35.6 us         35.5 us        19761        FFTW_C2R/700        1.88 us         1.88 us       373895
XTENSORFFTW_C2R/800        26.4 us         26.4 us        26465        FFTW_C2R/800        1.63 us         1.63 us       428799
XTENSORFFTW_C2R/900        28.4 us         28.4 us        24557        FFTW_C2R/900        1.80 us         1.80 us       389424
XTENSORFFTW_C2R/1000       38.0 us         38.0 us        18496        FFTW_C2R/1000       2.13 us         2.13 us       328994
XTENSORFFTW_C2C/100        6.64 us         6.64 us       105505        FFTW_C2C/100       0.210 us        0.210 us      3202304
XTENSORFFTW_C2C/200        7.58 us         7.58 us        92472        FFTW_C2C/200       0.433 us        0.433 us      1613302
XTENSORFFTW_C2C/300        15.3 us         15.3 us        45826        FFTW_C2C/300        1.14 us         1.14 us       616312
XTENSORFFTW_C2C/400        9.03 us         9.03 us        77678        FFTW_C2C/400       0.895 us        0.894 us       782228
XTENSORFFTW_C2C/500        17.2 us         17.2 us        40683        FFTW_C2C/500        1.85 us         1.85 us       379290
XTENSORFFTW_C2C/600        17.2 us         17.2 us        40816        FFTW_C2C/600        2.38 us         2.38 us       296109
XTENSORFFTW_C2C/700        17.5 us         17.5 us        40012        FFTW_C2C/700        2.82 us         2.82 us       254188
XTENSORFFTW_C2C/800        11.1 us         11.1 us        63042        FFTW_C2C/800        2.48 us         2.48 us       281086
XTENSORFFTW_C2C/900        28.6 us         28.6 us        24449        FFTW_C2C/900        5.33 us         5.33 us       130325
XTENSORFFTW_C2C/1000       20.0 us         20.0 us        35016        FFTW_C2C/1000       4.35 us         4.35 us       160745
```
