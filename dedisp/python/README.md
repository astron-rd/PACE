# FDD Python

## Usage

To simulate a signal, use `fdd-sim`:

```sh
uv run fdd-sim signal.h5
```

To run the FDD algorithm and store the result in a HDF5 file (`fddresult.h5`):

```sh
uv run fdd signal.h5
```

By default, it stores the result in a HDF5 file called `fdd.h5`. To plot the
dynamic spectrum, dedispersed signal, and the DM search space, use `fdd-plot`:

```sh
uv run fdd --dynspec signal.h5 --result fdd.h5 --image burst.png
```

### Linting

```sh
pre-commit run --all
```
