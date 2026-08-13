# Code examples for the ***PACE*** project

This repository provides simplified reference implementations from the radio
astronomy domain. PACE stands for Post-Correlation Acceleration for Astronomical
Data Processing Efficiency. It focuses on key computational patterns for the
following applications:

- Image-Domain Gridding (IDG)
- All-sky Imaging (LOFTY)
- Fourier-Domain Dedispersion (FDD)

Each application is implemented in one or more languages/frameworks to support
comparison, experimentation, benchmarking, and learning.

## Project documentation

This project is also published on Read the Docs at:

https://astron-pace.readthedocs.io/

The Read the Docs landing page is generated from `docs/index.md`.

## Structure

Each folder corresponds to a different application or framework/language
combination:

```
`idg/python` # Image-Domain Gridding in Python
`idg/cpp` # Image-Domain Gridding in C++
`idg/rust` # Image-Domain Gridding in Rust
```

## Purpose

The code in this repository may contain simplified versions of production code
to demonstrate concepts and usage patterns. It is **not production-ready**, but
rather a playground for experimentation, benchmarking, and learning.

## Usage

1. Navigate to the folder of the language/framework you want to explore.
1. Follow the instructions in the folder-specific `README` (if available).
1. Run or build the application as described.
