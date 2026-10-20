# Add a JAX backend to idg/python

## Why

First we checked that the numba implementation is correct, because it has no reference implementation to compare against.
The output matches the C++ grid within float rounding: max absolute difference about 1.5e-2 on a peak near 256, that is about 6e-6 relative to the peak.
So numba is good, fortunately, and we keep it as the reference and the default backend.

On top of that we decided to add a JAX backend as an alternative implementation of the same gridding kernels.
The two share all driver code and expose the same interface, so either can be selected at runtime.
The goal is a portable vectorized variant, not to replace numba.

## What changed

- Moved the numba kernels to `src/idg/backends/numba.py` and added `src/idg/backends/jax.py`.
- Added a backend registry in `src/idg/backends/__init__.py` with `get_backend(name)` and `available_backends()`.
- Unified the package: gridder, taper, config, main and types are shared once, only the kernels differ.
- Added a `--backend {numba,jax}` CLI option, defaulting to numba.
- Parallelized the JAX gridding with `jax.vmap` over batches of subgrids, so each batch is one jitted call that XLA threads across cores.
  Timesteps are padded to the batch maximum and masked, because subgrids have varying numbers of timesteps.
- Replaced the per-subgrid grid accumulate with one `jax.lax.scan` over all subgrids (`add_subgrids_to_grid`), so the grid stays in XLA buffers instead of round-tripping to the host NumPy array once per subgrid (CPU-only jax).
- Tests in `tests/test_backends.py` run both backends through the same parametrized checks.

## Notes

- The JAX gridding is only slightly faster than the serial JAX version (96 s to 92 s for the 10830-subgrid input at 32 threads on an Intel node).
  The vmap batching avoids the per-subgrid jit dispatch overhead, but the padded fast path dominates. This phase is not where jax wins.
- The big win is the grid accumulate: the old per-subgrid jitted add took 5.3 s, the `lax.scan` version takes 0.16 s, about 33x faster,
  with bit-identical output.
- Compared to numba the result is hardware dependent. On Intel nodes (haswell, skylake) jax gridding is 1.6 to 2.8x faster than numba.
  On AMD Zen2 nodes with many cores numba's prange scales better and is 1.5 to 2.0x faster than jax. So neither backend is universally faster,
  which is why numba stays the default.
- The `lax.scan` and `fori_loop` primitives are a win for the grid accumulate only. Rewriting the gridding kernel itself as a
  `lax.scan` over timesteps is about 10x slower, so the einsum array-notation form is kept.
- Correctness of the JAX backend is unchanged from the serial version and verified against C++ and numba.

## Validation

- pytest passes for both backends.
- JAX output vs C++ maxabs 0.01428, JAX vs numba maxabs 0.00394, both float-level.
- jax vmap-batched grid vs jax serial grid max diff 1.8e-4, float-level on a 3e8-magnitude sum.
