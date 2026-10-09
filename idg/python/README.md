# IDG in Python

Image-Domain Gridding (IDG) implemented in Python, with a selectable kernel
backend. Two backends are provided, exposing the same interface:

- `numba`: Numba JIT kernels, parallelized over subgrids with `prange`.
- `jax`: JAX kernels (currently a serial per-subgrid implementation).

Both produce the same numerical results (float32-level agreement with the
C++ implementation).

## Usage

Select the backend with `--backend`:

```bash
uv run idg {input-file} --backend numba
uv run idg {input-file} --backend jax
```

The default backend is `numba`. To store the resulting grid and subgrids as
`output.h5`:

```bash
uv run idg {input-file} --store
```

To write timings to a JSON file:

```bash
uv run idg {input-file} --json
# or a custom filename
uv run idg {input-file} --json custom.json
```

## Structure

```
idg/
├── pyproject.toml              # idg-python package definition
├── src/idg_python/
│   ├── config.py               # CLI / settings (backend selection)
│   ├── gridder.py              # Gridder driver (grid, add subgrids, transform)
│   ├── taper.py                # taper dispatch
│   ├── types.py                # shared constants
│   ├── main.py                 # entry point
│   └── backends/
│       ├── __init__.py         # get_backend() registry
│       ├── numba.py            # Numba kernels
│       └── jax.py              # JAX kernels
└── tests/
    └── test_backends.py
```

Each backend module exposes the same interface:
`evaluate_spheroidal`, `get_taper`, `visibilities_to_subgrids`,
`add_subgrid_to_grid` and `compute_phasor`.

## Environment

Requires Python >= 3.12. Set up with:

```bash
uv sync --dev
```

## Validation

Correctness is validated against the C++ implementation: max absolute grid
difference of `0.015028041`, relative-of-peak of `5.857762e-06` (float32-level
agreement).
