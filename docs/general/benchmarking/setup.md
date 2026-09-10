# Benchmarking setup for PACE

Bencher, a hosted continuous-benchmarking service, has been on trial for
recording the benchmark results of PACE: the
[framework evaluation](frameworks.md) selected it, a proof of concept uploaded
IDG Python timings with the [upload script](bencher/index.md), and a CI trial
ran the pytest-benchmark and criterion micro-benchmarks through it on
GitHub-hosted runners.

Access to the project on bencher.dev was a problem for part of the team, and the
hosted approach has further problems: GitHub runners are shared virtual
machines, too noisy for regression thresholds.

The recommendation is to run the benchmarks on the DAS-6 cluster with ReFrame as
the driver, started from GitHub Actions. Every implementation writes a result
file in a format defined by PACE, results are kept in git, and plots are
rendered into the docs site.

## Problems with Bencher

- **Access.** Not every team member could get access to the project on
  bencher.dev.
- **Platform churn.** Bencher's docs: "Do not specify an exact version if using
  Bencher Cloud as there are still occasional breaking changes." Five of the
  eight releases between May and July 2026 are marked BREAKING.
- **Runner noise.** A hosted runner is a virtual machine on shared hardware, so
  the CPU model and the load next to it change from run to run: `ubuntu-latest`
  varies 10 to 20 % run to run by github-action-benchmark's own estimate, and
  the Bencher trial on it raised a false +10.66 % alert on a 20 us kernel
  ([report](https://bencher.dev/perf/astron-pace/reports/f3bf8381-e065-4ae8-86e2-bd20cd186d2c)).
- **No GPUs.** The GPU and multi-node work of the later PACE milestones (M3 to
  M5) needs GPU nodes, which CI runners do not offer.
- **Data model.** Bencher, like github-action-benchmark, CodSpeed and Nyrkio,
  records a value per branch, testbed and commit. Encoding the language as
  "branch" only gives one timeline per language.

## Requirements

Beyond the [earlier criteria](frameworks.md), two requirements matter:

- **Comparison across implementations**: results for the same application in
  Python, C++, Rust, Julia and soon the OpenMP, OpenACC and GPU variants must be
  viewable side by side.
- **Low maintenance**: running a benchmark server is outside the scope of the
  project. DAS-6 is the way around that, but it should not turn into support
  requests for the people who maintain it.

## Proposed architecture

The setup consists of four layers.

1. **Emit**: every implementation writes one result file per run.
1. **Run**: ReFrame submits one Slurm job per application and implementation.
1. **Store**: result files are committed to git under `results/`.
1. **View**: a script renders comparison and scaling plots into the docs site.

The time measurements are self-reported by the applications, split into phases
and reported in seconds. A result file records the execution environment and
software version (e.g. commit id). For example:

```json
{
  "application": "idg",
  "implementation": "rust",
  "commit": "52cd29a",
  "testbed": { "host": "node503", "cpu": "2x AMD EPYC 7302", "gpu": "RTX A4000" },
  "parameters": { "grid_size": 4096, "subgrid_size": 32 },
  "timings_s": {
    "compile": 0.0, "grid": 1.203, "ifft": 0.311, "add": 0.087, "transform": 0.402
  }
}
```

Currently, only IDG Python writes a result file. The other implementations print
phase times to stdout.

## Execution environment

The benchmarks run on the DAS-6 Slurm cluster. While PACE has budget for
dedicated infrastructure, reusing existing DAS-6 resources is the most pragmatic
approach given current constraints. Two properties of the cluster determine the
choice of tooling:

- Jobs are capped at 15 minutes during working hours, so an experiment has to be
  many short jobs rather than one long sweep.
- Compute nodes are only reachable through Slurm jobs, so the tool has to submit
  jobs instead of running the benchmarks where it is started.

Triggered by GitHub Actions, a runner on the DAS-6 control node starts ReFrame,
which submits the jobs with `sbatch` and collects the results while the workflow
keeps the logs.

## Candidate tools

| Tool         | Cluster   | Notes                                                                                                                                                                                                      |
| ------------ | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ReFrame      | native    | Knows Slurm partitions and writes the job scripts, runs every combination of `parameter()` values as separate jobs, keeps results in SQLite and compares them across sessions.                             |
| slurm-action | srun      | One workflow step becomes one job from a self-hosted runner on the control node. Parameterisation, result collection and comparison are hand-written in the workflow.                                      |
| JUBE         | templates | From the Juelich Supercomputing Centre (JSC). Submits through job templates, runs every combination of parameterset values, collects results with regex patterns into CSV tables. Unmaintained since 2024. |

ReFrame is the driver: it submits to Slurm natively and is easy to install via
`uv`. `slurm-action` is the fallback if ReFrame does not work out on DAS-6: it
submits to Slurm, but parameterisation, result collection and comparison are
hand-written. JUBE is ruled out because it is unmaintained.

Also considered: ReBench (no Slurm support, one long run on a dedicated
machine), hyperfine (no Slurm support, times the whole process), Ramble and
Benchpark (expect Spack-built applications), Pavilion2 (system acceptance tests
rather than performance studies), Conbench (needs a server) and asv (single
Python project per commit).

## Next steps

1. Define the JSON specification of the result file.
1. Make every implementation write a result file.
1. Run the benchmarks on DAS-6 from GitHub Actions.
1. Render the comparison and scaling plots into the docs site.

## Sources

- Bencher: [changelog](https://bencher.dev/docs/reference/changelog/),
  [CLI install](https://bencher.dev/docs/how-to/install-cli/)
- github-action-benchmark:
  [repository](https://github.com/benchmark-action/github-action-benchmark)
- CodSpeed:
  [benchmarks in CI without noise](https://codspeed.io/blog/benchmarks-in-ci-without-noise)
- Nyrkio: [repository](https://github.com/nyrkio/nyrkio)
- DAS-6: [job policy](https://www.cs.vu.nl/das/jobs.shtml)
- ReFrame:
  [tutorial](https://reframe-hpc.readthedocs.io/en/stable/tutorial.html),
  [manpage](https://reframe-hpc.readthedocs.io/en/stable/manpage.html)
- slurm-action: [repository](https://github.com/astron-rd/slurm-action)
- JUBE: [repository](https://github.com/FZJ-JSC/JUBE),
  [tutorial](https://apps.fz-juelich.de/jsc/jube/docu/tutorial.html)
- ReBench: [configuration](https://rebench.readthedocs.io/en/latest/config/)
- hyperfine: [repository](https://github.com/sharkdp/hyperfine)
- Ramble:
  [getting started](https://ramble.readthedocs.io/en/latest/getting_started.html)
- Benchpark: [repository](https://github.com/llnl/benchpark)
- Pavilion2: [documentation](https://pavilion2.readthedocs.io/en/latest/)
- Conbench: [repository](https://github.com/conbench/conbench)
