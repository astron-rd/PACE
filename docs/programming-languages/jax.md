Jax is a framework that can perform many tasks on data. These tasks can be
performed using an interface that is identical to Numpy making it very
attractive.

Read: https://docs.jax.dev/en/latest/jax-101.html to get started

In short Jax consist of three parts:

1. Jit: https://docs.jax.dev/en/latest/\_autosummary/jax.jit.html
1. vmap: https://docs.jax.dev/en/latest/\_autosummary/jax.vmap.html#jax.vmap
1. autograd: https://docs.jax.dev/en/latest/automatic-differentiation.html

Jit; Parallelize and optimize entire functions for use on CPU or accelerators
using OpenXLA. Vmap; Vectorize calls across a multi dimensional array axis
Autograd; Differentiate / find gradients

### However, there are some important caveats to using Jax, below a short summary:

1. Output data is immutable, you can't `data[12] = y`. Instead you could (but
   shouldn't) `data.at(12) = y` Each time you do this its a full memory copy of
   the data with one element changed so it has **_HORRENDOUS_** performance if
   you use this.
1. Data sizes can not be dynamic, array allocations can not be based on the
   result of dynamic variables within a `jitted` function. They must be known at
   call time.
1. Control flow is not supported within `jitted` functions on dynamic variables.
1. Jax runs at 32bit precision no matter the datatype

For more info look here:
https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html

### Overcoming the caveats

1. Use builtin Numpy operators like `map` or `vectorize` to prevent having to
   mutate individual elements
1. Hoist the allocation of of arrays based on dynamic variables outside of the
   jitted function and pass the dynamically sized arrays as arguments
1. Use `static_argnums` as argument on the `@jit` call to help the jitter
   identify the dynamic variables
1. Specify 64bit mode using:
   https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
