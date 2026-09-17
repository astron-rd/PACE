# XTensor

## Overview

[*xtensor*](https://xtensor.readthedocs.io/en/latest/) is a C++ library for working with multi-dimensional arrays, inspired by [NumPy](https://numpy.org/). This involves the basic *N*-dimensional structures, but also an entire expression engine that allows numerical computations on any object that implements the expression interface, such as the containers; these expressions are lazy.

The library also comes with *adapters* to convert, say, C-style arrays or standard library vectors, into their expression system. ... something about NumPy bindings...

Moreover, they provide a lot of convenient functions, such as initializers, slicing, concatenation, rearranging elements, reducers, NaN functions, and much more. For readers familiar with NumPy, the following page ["From NumPy to xtensor"](https://xtensor.readthedocs.io/en/latest/numpy.html), describes a lot of these functions and show the NumPy counterparts.

## The Whole Family

*xtensor* is but a small part of the whole [*xtensor* stack](https://github.com/xtensor-stack), which consits of of various libraries that extend *xtensor*, such as:

- *xsimd* for working with SIMD intrinsics,
- *xtensor-fftw* for performing FFTs using [FFTW](https://fftw.org/),
- *xtensor-blas* for linear algebra (similar to NumPy's `linalg`),
- *xtensor-io* for reading common file formats,
- *xtensor-python* for using NumPy data structures from C++

Of this family, we used `xtensor`, `xtensor-fftw`, and briefly investigated `xtensor-io` for reading binary `.npy` files.

## The Verdict
