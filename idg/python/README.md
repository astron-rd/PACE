# Python IDG implementation

This is a Python + Numba implementation of IDG. It takes visibilities in an input file and performs Image-Domain Gridding on them to create an image.

For more information on the input data format, see the [Data Format documentation](../../docs/general/data-format.md).

## Parallelization

The Python implementation uses Numba's Just-In-Time (JIT) compilation to accelerate the computational kernels. Parallelization is implemented using Numba's `parallel=True` decorator and the `prange` function.

In the `visibilities_to_subgrids` kernel, the processing of subgrids is distributed across CPU cores:

```python
@nb.njit(parallel=True)
def visibilities_to_subgrids(...):
    # ...
    for s in nb.prange(nr_subgrids):
        visibilities_to_subgrid(...)
```

- **`parallel=True`**: This tells Numba to attempt to automatically parallelize loops and optimize the code for multi-core execution.
- **`nb.prange`**: This is a special version of the Python `range` function that explicitly tells Numba that the loop iterations are independent and can be executed in parallel.

Similar to the C++ version, the gridding process is the main computational bottleneck, and parallelizing this stage provides the most significant speedup.

## Basic Usage

```sh
uv run idg {input-file}
```

You can get an input file from the input generator in `idg/input`. Use the `--store` flag to store the resulting image in an HDF5 file.

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
