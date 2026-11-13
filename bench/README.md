# Benchmarking

Benchmark tooling for PACE. The result file specification is defined with Pydantic.
The generated documentation is in [`4-specification.md`](../docs/general/benchmarking/4-specification.md).

## Updating the specification

1. Edit the model in [`src/pace_bench/models.py`](src/pace_bench/models.py).

1. Bump the specification version, using `major`, `minor` or `patch` depending on the change:

    ```sh
    uv version --bump minor
    ```

1. Update the documentation:

    ```sh
    uv run export
    ```

> [!NOTE]
> See the [Pydantic documentation on models](https://pydantic.dev/docs/validation/latest/concepts/models/) for the available field types and constraints.

## Recording a result

The `record` command turns the parameters and timings of a run into a result file. It adds the version, commit and hardware, validates the result and stores it under `results/`:

```sh
uv run --project bench record <measurements.json> \
    --application <application> --language <language> --timestamp <start time>
```

Run it inside the job that ran the benchmark, as it reads the hardware of that node. The [benchmark workflow](../.github/workflows/benchmark.yaml) runs it on DAS-6.
