# Benchmarking setup for PACE

Bencher, a hosted continuous-benchmarking service, has been on trial for
recording the benchmark results of PACE: the
[framework evaluation](frameworks.md) selected it, a proof of concept uploaded
IDG Python timings with the [upload script](bencher/index.md), and a CI trial
ran the pytest-benchmark and criterion micro-benchmarks through it on
GitHub-hosted runners.

Access to the project on bencher.dev was a problem for part of the team, and the
hosted approach has further problems: GitHub runners are shared virtual
machines, too noisy for regression thresholds, though
[`slurm-action`](https://github.com/astron-rd/slurm-action) can move the run
itself onto a Slurm cluster. Of the candidates surveyed, ReFrame and JUBE suit a
Slurm cluster and ReBench a dedicated machine. The proposal is a result format
defined within PACE that every implementation writes, a driver that runs every
combination of settings, results kept in git, and plots rendered into the docs
site.

## Problems with the trial

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
  No storage tool fixes this. The GPU and multi-node work of the later PACE
  milestones (M3 to M5) will need GPU nodes, which CI runners do not offer.
- **Data model.** Bencher, like github-action-benchmark, CodSpeed and Nyrkio,
  records a value per branch, testbed and commit. Encoding the language as
  "branch" only gives one timeline per language.

## Requirements

Beyond the [earlier criteria](frameworks.md), two requirements matter:

- **Comparison across implementations**: results for the same application in
  Python, C++, Rust, Julia and soon the OpenMP, OpenACC and GPU variants must be
  viewable side by side.
- **Low maintenance**: running a benchmark server is outside the scope of the
  project. Existing infrastructure such as DAS-6 is the way around that, but it
  should not turn into support requests for the people who maintain it.

## Proposed architecture

The setup consists of four layers.

1. **Emit**: every implementation writes one result file per run.
1. **Run**: a runner starts a benchmark run for every implementation of the
   various applications, directly or as a Slurm job. It can be an existing
   benchmarking tool or a shell script.
1. **Store**: result files committed under `results/`.
1. **View**: a script renders comparison and scaling plots into this
   documentation site.

A measurement is the per-phase wall-clock time from the application's own
timers. Warm-up, compilation or JIT time is reported as a phase of its own. A
result file records the hardware and the commit id. For example:

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

Currently, only IDG Python writes JSON. The C++, Rust, Julia and FDD mains print
phase times to stdout under their own labels, and all-sky times whole runs from
its pytest benchmarks.

## Execution environments

Two kinds of environment are in reach and they favour different tooling. A
**dedicated machine** (a workstation or a reserved server) is the simplest: no
scheduler, no time limits, root available for pinning CPU frequencies. A **Slurm
cluster** gives access to the GPUs that the GPU-offloading milestone (M3) needs.
If the cluster route is taken, three properties matter for the choice of
tooling:

- A job gets its nodes to itself, which removes the noise problem of hosted CI.
- Jobs are capped at 15 minutes during working hours, so an experiment has to be
  many short jobs rather than one long sweep.
- The driver has to submit to Slurm, which excludes CI-only tools.

GitHub Actions can stay the trigger:
[`slurm-action`](https://github.com/astron-rd/slurm-action) runs a workflow step
through `srun` from a self-hosted runner on the control node, so the measurement
happens on a cluster node while the workflow keeps the logs.

While PACE has budget for dedicated infrastructure, reusing existing DAS-6
resources is the most pragmatic approach given current constraints.

## Candidate tools

| Tool         | Cluster   | Notes                                                                                                                                                                                            |
| ------------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ReFrame      | native    | Knows Slurm partitions and writes the job scripts, runs every combination of `parameter()` values as separate jobs, keeps results in SQLite and compares them across sessions.                   |
| JUBE (JSC)   | templates | Submits through job templates, runs every combination of parameterset values, collects results with regex patterns into CSV tables. Not on PyPI, last release May 2024.                          |
| ReBench      | no        | Config lists implementations x benchmarks x input sizes directly. One config is one long local run and denoising needs root, so it suits a dedicated machine, not a time-capped cluster.         |
| hyperfine    | no        | Repeats a command with warm-up and varies one setting at a time, per-run times to JSON. Times the whole process, so it adds nothing to the applications' own phase timers, but needs zero setup. |
| Shell script | sbatch    | Works anywhere, you write the loop. Reimplements what the drivers already do (parameterisation, result collection, comparison).                                                                  |

Also considered: Ramble/Benchpark (built for standard benchmark suites, expects
Spack-built applications), Pavilion2 (system acceptance tests rather than
performance studies), Bencher self-hosted and github-action-benchmark (no
Slurm), Nyrkio and Conbench (need a server and model results as a commit
timeline), CodSpeed (SaaS, simulated CPU, no GPU/Julia), asv (single Python
project per commit).

## Open questions

- Where do the benchmarks run: a dedicated machine, the DAS-6 nodes, another
  Slurm cluster, or a mix?
- Which driver follows from that: ReFrame, JUBE, ReBench, hyperfine or a script?

## Sources

- [DAS-6 job policy](https://www.cs.vu.nl/das/jobs.shtml)
- ReFrame:
  [tutorial](https://reframe-hpc.readthedocs.io/en/stable/tutorial.html),
  [manpage](https://reframe-hpc.readthedocs.io/en/stable/manpage.html)
- [JUBE](https://github.com/FZJ-JSC/JUBE),
  [JUBE tutorial](https://apps.fz-juelich.de/jsc/jube/docu/tutorial.html)
- [Ramble](https://ramble.readthedocs.io/en/latest/getting_started.html),
  [Benchpark](https://github.com/llnl/benchpark),
  [Pavilion2](https://pavilion2.readthedocs.io/en/latest/),
  [ReBench config](https://rebench.readthedocs.io/en/latest/config/),
  [hyperfine](https://github.com/sharkdp/hyperfine)
- [Bencher changelog](https://bencher.dev/docs/reference/changelog/),
  [install docs](https://bencher.dev/docs/how-to/install-cli/),
  [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark),
  [Nyrkio](https://github.com/nyrkio/nyrkio),
  [CodSpeed noise measurements](https://codspeed.io/blog/benchmarks-in-ci-without-noise),
  [Conbench](https://github.com/conbench/conbench)
