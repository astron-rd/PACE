# Julia IDG implementation

This is a Julia implementation of Image Domain Gridder. The program transforms visibilities into an image using the Image-Domain Gridding (IDG) algorithm. Input can be loaded from an HDF5 file.

## About Julia

Julia is a dynamically JIT-compiled language that focuses on being useful for scientific computing while being very performant. The main way most users interact with Julia is using the Julia REPL.

Julia gets its speed from acting like an interpreter, but JIT-compiling your functions right before execution. This is complicated by Julia's dynamic-ish typing system which requires the compiler to account for uncertainties in the types used by the functions. The best way to create performant Julia code is to be as explicit as possible about the types being used. Learn more about Julia from [the documentation](https://docs.julialang.org/en/v1/#man-introduction).

## Prerequisites

To run Julia programs you will need the Julia compiler. Julia toolchains are managed using the `juliaup` tool. You can use these [installation instructions](https://docs.julialang.org/en/v1/manual/installation/).

Generate an input dataset using the input generator in `idg/input`.

## Running

1. Place `input.h5` in the working directory
2. Install and precompile the dependencies:
```
julia --project=. -e "using Pkg; Pkg.precompile()"
```
Precompiling packages is somewhat unreliable on Julia, so this may fail randomly. In that case try running the command again.

3. Run IDG with the following command:

```
julia --project=. -t auto main.jl
```

`--project=.` selects the current directory as the environment for the interpreter. `-t auto` tells julia to spawn the same number of threads as CPU cores which are available.
