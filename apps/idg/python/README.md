# Python implementation of Image-Domain Gridding (IDG)

This is a Python implementation of IDG. It takes visibilities in an input file and performs Image-Domain Gridding on them to create an image. The computational kernels are provided by two interchangeable backends, selected at runtime with `--backend`:

- `numba`: Numba JIT kernels (default, reference implementation).
- `jax`: JAX kernels, an array-notation variant of the same math.

Both produce the same numerical results (float32-level agreement with the C++ implementation).

For more information on the input data format, see the [Data Format documentation](../../../docs/general/data-format.md).

## Parallelization

The computational kernels are parallelized over subgrids, which is the main bottleneck, similar to the C++ version.

The Numba backend uses Just-In-Time (JIT) compilation with `parallel=True` and `nb.prange`, which tells Numba that the loop iterations over subgrids are independent and can be executed in parallel on the CPU cores:

```python
@nb.njit(parallel=True)
def visibilities_to_subgrids(...):
    # ...
    for s in nb.prange(nr_subgrids):
        visibilities_to_subgrid(...)
```

The JAX backend batches subgrids along a leading axis and parallelizes over that axis with `jax.vmap`. Each batch is one jitted call that XLA threads across cores. Adding subgrids to the grid runs as a single `jax.lax.scan` so the grid stays in XLA buffers instead of round-tripping to the host per subgrid.

## Performance

Both backends are correct and interchangeable. Neither is universally faster. Numba scales better on high-core-count AMD nodes, while JAX is faster on Intel nodes, so the best backend depends on the hardware.

## Basic Usage

```sh
uv run idg {input-file}
```

Select the backend with `--backend` (default: `numba`):

```sh
uv run idg {input-file} --backend jax
```

Use the `--store` flag to store the resulting image in an HDF5 file and the `--json` flag to write timings to a JSON file.

You can get an input file from the input generator in `apps/idg/input`.

### Unit tests

```sh
uvx pre-commit run --hook-stage manual --all -v pytest-idg
```

### Linting

```sh
uvx pre-commit run --all
```

### Packaging

```sh
uvx pre-commit run --hook-stage manual --all -v build-idg
```

## Structure

```
apps/idg/python/
├── pyproject.toml              # idg package definition
├── src/idg/
│   ├── config.py               # CLI / settings (backend selection)
│   ├── gridder.py              # Gridder driver (grid, add subgrids, transform)
│   ├── types.py                # shared constants
│   ├── main.py                 # entry point
│   └── backends/
│       ├── __init__.py         # get_backend() registry
│       ├── numba.py            # Numba kernels
│       └── jax.py              # JAX kernels
└── tests/
```

Each backend module exposes the same interface: `evaluate_spheroidal`, `get_taper`, `visibilities_to_subgrids`, `add_subgrids_to_grid` and `compute_phasor`.
