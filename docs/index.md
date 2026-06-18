# Introduction

PACE provides simplified reference applications from the radio astronomy domain.
The goal is not to reproduce full production pipelines, but to capture the key
computational patterns that make these applications interesting for evaluating
programming languages and parallelisation or acceleration strategies.

This documentation is published on Read the Docs at:

https://astron-pace.readthedocs.io/

The top-level repository `README.md` is a brief summary, the main documentation is
hosted here.

The current PACE applications are:

- Image-Domain Gridding (IDG)
- All-sky Imaging (LOFTY)
- Fourier-Domain Dedispersion (FDD/dedisp)

## Documentation structure

The documentation is organized into three categories:

- Applications
  - Image-Domain Gridding
  - All-sky Imaging (LOFTY)
  - Fourier-Domain Dedispersion
  - Fourier-Domain Dedispersion Details
- Programming Languages
  - Jax
  - Python
  - Rust
- General
  - Benchmarking Frameworks
  - Data Format

Use the navigation menu to browse these sections.

## Repository layout

Each top-level folder contains code for a particular application, language, or
framework combination. For example:

- `idg/python` — Image-Domain Gridding in Python
- `idg/cpp` — Image-Domain Gridding in C++

## Usage

1. Navigate to the folder for the language/framework you want to explore.
2. Follow the instructions in the folder-specific `README` if available.
3. Run or build the application as described there.
