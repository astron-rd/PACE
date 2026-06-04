# Overview
This document outlines the evaluation and selection process for a benchmarking framework to support the PACE project's performance tracking needs across multiple programming languages and systems.

# The challenge

PACE requires a robust solution for recording, storing, and visualizing benchmarking results that can handle:
- Experiment-level data: parameters, configurations, and custom metrics such as fine-grained timings
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

Bencher has the best score overall and offers the follow key advantages:

* Cross-Language Support
  * Native Rust SDK
  * Google Benchmark/Catch2 integration for C++
  * Python support via custom benchmarks
  * Language-agnostic JSON adapter

* Continuous Performance Benchmarking
  * Built-in regression detection
  * CI/CD integration capabilities
  * Historical trend analysis

* Flexibility
  * Self-hosted or cloud deployment
  * Custom metrics and parameters
  * Adaptable to various benchmarking scenarios

* Team Collaboration
  * Shared dashboards and reports
  * Performance regression alerts
  * Comprehensive visualization

# Proof of concept
We successfully implemented a Bencher integration, demonstrating:

## Technical implementation
* IDG Python refactoring: Modified to support benchmarking metrics output
* Custom upload script: Handles JSON result formatting and Bencher API communication
* Flexible metric tracking: Supports both timing and parameter data

## Current approach
We're using Bencher's "custom benchmarks" mode, which provides:
- Unlimited metric flexibility beyond simple runtime tracking
- JSON-based result reporting
- Custom parameter and metadata inclusion
- Hostname, git hash, and timestamp tracking

Example integration
```
python3 bencher_upload.py --project $BENCHER_PROJECT \
                         --benchmark idg-cpp \
                         --branch main \
                         --token $BENCHER_API_TOKEN \
                         results.json
```

So far, we have completed:
- Framework evaluation and selection
- Proof of concept implementation
- IDG Python benchmarking integration
- Custom upload script development

The following is future work:
- Gain experience: Continue using Bencher with current benchmarks
- Define metrics: Finalize standard metrics for all components
- CI integration: Automate benchmarking in pull requests
- Expand coverage: Include C++ and Rust components

## Conclusion
Bencher emerged as the optimal solution due to its unique combination of continuous performance tracking, multi-language support, and flexibility. The successful proof of concept confirms its suitability for PACE's diverse benchmarking requirements across different programming languages and system components.

The custom benchmarks approach, while requiring JSON output from applications, provides the necessary flexibility to track both high-level performance metrics and detailed system parameters, making it well-suited for PACE's benchmarking needs.

## References
- https://github.com/astron-rd/PACE/issues/7
