# XTensor

## Overview

[xtensor](https://xtensor.readthedocs.io/en/latest/) is a C++ library for working with multi-dimensional arrays, inspired by [NumPy](https://numpy.org/). This involves the basic *N*-dimensional structures, but also an entire expression engine that allows numerical computations on any object that implements the expression interface, such as the containers; these expressions are lazy.

The library also comes with *adapters* to convert, say, C-style arrays or standard library vectors, into their expression system.

Moreover, they provide a lot of convenient functions, such as initializers, slicing, concatenation, rearranging elements, reducers, NaN functions, and much more. For readers familiar with NumPy, the following page ["From NumPy to xtensor"](https://xtensor.readthedocs.io/en/latest/numpy.html), describes a lot of these functions and show the NumPy counterparts.

### The whole family

xtensor is but a small part of the whole [*xtensor* stack](https://github.com/xtensor-stack), which consits of of various libraries that extend xtensor, such as:

- xsimd for working with SIMD intrinsics,
- xtensor-fftw for performing FFTs using [FFTW](https://fftw.org/),
- xtensor-blas for linear algebra (similar to NumPy's `linalg`),
- xtensor-io for reading common file formats,
- xtensor-python for using NumPy data structures from C++.

Of this family, we used xtensor, xtensor-fftw, and briefly investigated xtensor-io for reading binary `.npy` files.

On top of that, this repository uses [xtensor-wrappers](https://github.com/astron-rd/xtensor-wrappers), a small header-only library that provides a reusable, plan-based FFTW API around xtensor-fftw, including higher-level wrappers for 2D and batched transforms.

## The good, the bad, and the ugly

xtensor offers an incredibly convenient way to work with multi-dimensional vectors, especially if you're introduction to scientific computing is with Python / NumPy. Defining *N*-dimensional arrays and manipulating them feels very natural, espcially, if you compare it to using C++'s standard library `std::vector`, and many operators that would require (multiple levels of) loops, are possible with a single function call. Thus, code that uses *xtensor* tends to be easier to read.

In particular, this is the case for xtensor-fftw that wraps the commonly used FFTW3 library for computing the FFT; an operation that is omnipresent in radio astronomy. FFTW3 is written in portable C, thus offers a (rather archaic) C-style interface, which requires a lot of manual memory management and can therefore be quite error prone. However, xtensor-fftw isn't as mature as xtensor itself and isn't very actively developed, although it seems they do keep it up-to-date. They, for example, lack some convenience functions that one might expect coming from NumPy, such as a multi-dimensional `fftshift` operation.

Lastly, we investigated the use of xtensor-io for storing and loading data stored in the `.npy` format accross all languages, since most languages offer a library with such functionality. Sadly, it does not support compound data types, which means having to store members of, e.g., a `struct` in a different file.

## `xtensor-fftw` benchmark

The most common way to call xtensor-fftw is through its convenience functions, which create a new FFTW plan on every call. Planning is an expensive step, so comparing that against a benchmark that creates one plan up front and reuses it is apples versus oranges. Measured that naive way, FFTW3 appears up to a hundred times faster for small FFTs. When both sides create the plan once and reuse it, the difference almost completely disappears, as the table below shows. xtensor-fftw is a wrapper around FFTW, so once the plan exists, execution is the same code path.

| Operation | FFTW3 time | xtensor-fftw plan time | Speedup (FFTW3 over xtensor-fftw) |
| :--- | :---: | :---: | :---: |
| R2C/100 | 0.051 us | 0.052 us | $\\approx 1.0\\times$ |
| R2C/1000 | 0.714 us | 0.739 us | $\\approx 1.0\\times$ |
| C2R/100 | 0.054 us | 0.053 us | $\\approx 1.0\\times$ |
| C2R/1000 | 0.744 us | 0.752 us | $\\approx 1.0\\times$ |
| C2C/100 | 0.075 us | 0.081 us | $\\approx 1.0\\times$ |
| C2C/1000 | 1.67 us | 1.67 us | $\\approx 1.0\\times$ |

These values were obtained with the same [`fft-benchmark`](https://git.astron.nl/RD/fft-benchmark) tool, with the xtensor-fftw benchmark creating its plan before the timed loop, and the benchmarks ran on `node508` of the [DAS-6 cluster](https://www.cs.vu.nl/das6/clusters.shtml).

The takeaway is that the choice of wrapper matters little for performance. What matters is using the plan based interface, so plan creation happens once and the same plan is reused for every transform of the same size. That is where the real end-to-end gains come from, and it is exactly what `xtensor-wrappers` provides and what the C++ implementations in this repository use.
