# IDG in JAX

Image-Domain Gridding (IDG) implemented in Python with JAX. This is the JAX
equivalent of the `../numba` implementation and mirrors its structure and
results.

## Usage

```bash
uv run idg-jax {input-file}
```

To store the resulting grid and subgrids as `output.h5`:

```bash
uv run idg-jax {input-file} --store
```

To write timings to a JSON file:

```bash
uv run idg-jax {input-file} --json
# or a custom filename
uv run idg-jax {input-file} --json custom.json
```

## Structure

```
idg/jax/
├── pyproject.toml          # idg-jax package definition
├── src/idg_jax/
│   ├── config.py           # CLI / settings
│   ├── gridder.py          # Gridder driver (grid, add subgrids, transform)
│   ├── taper.py            # spheroidal taper
│   ├── types.py            # shared constants
│   ├── main.py             # entry point
│   └── kernels/
│       ├── gridding.py     # JAX gridding kernels
│       └── spheroidal.py   # spheroidal polynomial evaluation
└── tests/
    └── test_kernels.py
```

The JAX kernels in `kernels/gridding.py` use `jax.jit` and `jax.lax.scan` in
place of the numba `@njit` loops in `../numba`. FFTs are computed with
`numpy.fft` on the host, matching the numba implementation.

## Environment

Requires Python >= 3.12. Set up with:

```bash
uv sync --dev
```

## Validation

The correctness reference numbers match the numba and C++ implementations:
max absolute grid difference of `0.015028041`, relative-of-peak of
`5.857762e-06` (float32-level agreement).
