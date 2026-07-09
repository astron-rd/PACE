# Python IDG implementation

This is a Python + Numba implementation of IDG. It takes visibilities in an input file and performs Image-Domain Gridding on them to create an image.

## Basic Usage

```sh
uv run idg {input-file}
```

You can get an input file from the input generator in `idg/input`. Use the `--store` flag to store the resulting image in an HDF5 file.

### Unit tests

```sh
pre-commit run --hook-stage manual --all -v pytest-idg
```

### Linting

```sh
pre-commit run --all
```

### Packaging

```sh
pre-commit run --hook-stage manual --all -v build-idg
```
