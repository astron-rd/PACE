# Overview

This document outlines the evaluation and selection process for a benchmarking
framework to support the PACE project's performance tracking needs across
multiple programming languages and systems.

# The challenge

PACE requires a robust solution for recording, storing, and visualizing
benchmarking results that can handle:

- Experiment-level data: parameters, configurations, and custom metrics such as
  fine-grained timings
- System-level metrics: CPU, GPU, and resource utilization
- Multi-language support: C++, Rust, Python, and Julia
- Performance regression tracking: Monitoring changes across commits
- Team collaboration: Sharing results and insights

# Candidate evaluation

We evaluated six prominent tools against our requirements.

Scoring scheme:

- ✅ = 1 point
- ◯ = 0.5 point
- ❌ = 0 points

| Framework | Open source | Self-hosted | Multi-language | Performance tracking | Visualization |
|---|---|---|---|---|---|
| [Bencher](https://bencher.dev/) | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Neptune.ai](https://neptune.ai/) | ❌ | ✅ | ◯ | ❌ | ✅ |
| [MLflow](https://mlflow.org/) | ✅ | ✅ | ❌ | ❌ | ✅ |
| [Prometheus/Grafana](https://prometheus.io/) | ✅ | ✅ | ◯ | ❌ | ✅ |
| [DVC](https://github.com/treeverse/dvc/wiki/Debugging,-Profiling-and-Benchmarking-DVC) | ✅ | ✅ | ❌ | ❌ | ◯ |
| [Hyperfine](https://github.com/sharkdp/hyperfine) | ✅ | ❌ | ✅ | ❌ | ❌ |

## Conclusion

Bencher emerged as the optimal solution due to its unique combination of
continuous performance tracking, multi-language support, and flexibility.

## References

- https://github.com/astron-rd/PACE/issues/7
