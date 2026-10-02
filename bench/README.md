# Benchmarking

Benchmark tooling for PACE. The result file specification is defined with Pydantic.
The generated documentation is in [`4-specification.md`](../docs/general/benchmarking/4-specification.md).

## Updating the specification

1. Edit the model in [`src/pace_bench/models.py`](src/pace_bench/models.py). See the [Pydantic documentation on models](https://pydantic.dev/docs/validation/latest/concepts/models/) for the available field types and constraints.

1. Bump the specifications version, using `major`, `minor` or `patch` depending on the change:

   ```sh
   uv version --bump minor
   ```

1. Update the documentation:

   ```sh
   uv run export
   ```
