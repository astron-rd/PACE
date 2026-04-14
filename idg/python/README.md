# Idg Python

## Usage

```sh
uv run idg {arguments}
```

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
