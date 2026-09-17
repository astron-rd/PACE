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

## The good, the Bad, and the ugly

*xtensor* offers an incredibly convenient way to work with multi-dimensional vectors, especially if you're introduction to scientific computing is with Python / NumPy. Defining *N*-dimensional arrays and manipulating them feels very natural, espcially, if you compare it to using C++'s standard library *std::vector*, and many operators that would require (multiple levels of) loops, are possible with a single function call. Thus, code that uses *xtensor* tends to be easier to read.

In particular, this is the case for *xtensor-fftw* that wraps the commonly used *FFTW3* library for computing the FFT; an operation that is omnipresent in radio astronomy. FFTW3 is written in portable C, thus offers a (rather archaic) C-style interface, which requires a lot of manual memory management and can therefore be quite error prone. However, *xtensor-fftw* isn't as mature as *xtensor* itself and isn't very actively developed, although it seems they do keep it up-to-date. They, for example, lack some convenience functions that one might expect coming from NumPy, such as a multi-dimensional `fftshift` operation.

Lastly, we investigated the use of `xtensor-io` for storing and loading data stored in the `.npy` format accross all languages, since most languages offer a library with such functionality. Sadly, it does not support compound data types, which means having to store members of, e.g., a `struct` in different file.


## `xtensor-fftw` bechmark

TBD
