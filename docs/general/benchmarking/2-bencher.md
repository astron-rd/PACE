# Bencher

Bencher is the selected benchmarking framework for PACE, chosen for its unique combination of continuous performance tracking, multi-language support, and flexibility.

## Key advantages

- **Cross-language support**

  - Native Rust SDK
  - Google Benchmark/Catch2 integration for C++
  - Python support via custom benchmarks
  - Language-agnostic JSON adapter

- **Continuous performance benchmarking**

  - Built-in regression detection
  - CI/CD integration capabilities
  - Historical trend analysis

- **Flexibility**

  - Self-hosted or cloud deployment
  - Custom metrics and parameters
  - Adaptable to various benchmarking scenarios

- **Team collaboration**

  - Shared dashboards and reports
  - Performance regression alerts
  - Comprehensive visualization

## Proof of concept

We successfully implemented a Bencher integration, demonstrating:

### Technical implementation

- IDG Python refactoring: Modified to support benchmarking metrics output
- Custom upload script: Handles JSON result formatting and Bencher API
  communication
- Flexible metric tracking: Supports both timing and parameter data

### Current approach

We're using Bencher's "custom benchmarks" mode, which provides:

- Unlimited metric flexibility beyond simple runtime tracking
- JSON-based result reporting
- Custom parameter and metadata inclusion
- Hostname, git hash, and timestamp tracking

Example integration:

```
python3 bencher_upload.py --project $BENCHER_PROJECT \
                          --benchmark idg-cpp \
                          --branch main \
                          --token $BENCHER_API_TOKEN \
                          results.json
```

## Status and future work

While Bencher served as a proof of concept, the project has since evolved toward a custom workflow, described in [3-architecture.md](3-architecture.md).

Completed:

- Framework evaluation and selection
- Proof of concept implementation
- IDG Python benchmarking integration
- Custom upload script development

## Bencher upload script

The Bencher upload script is located at `bencher/upload.py` in the repository.

It uploads benchmarking results to Bencher (https://bencher.dev/) for
performance tracking and analysis.

### Prerequisites

Set the following environment variables:

```
export BENCHER_PROJECT="your-project-name"
export BENCHER_API_TOKEN="your-api-token"
```

### Usage

Upload benchmarking results from a JSON file:

```
python3 bencher/upload.py --project "$BENCHER_PROJECT" \
                          --token "$BENCHER_API_TOKEN" \
                          --benchmark idg \
                          --branch python \
                          results.json
```

#### Arguments

- `project`: Bencher UUID
- `token`: Bencher API token
- `benchmark`: The application under test, e.g. `idg`
- `branch`: The specific implementation of this benchmark, e.g. `python` (not
  the Git branch name)
- `results.json`: Path to JSON file with benchmarking results

### Successful output

Upon successful upload, you should see output similar to:

```
Successfully uploaded X metrics to Bencher
Project: [project], Branch: [Git branch], Hash: [Git hash], Testbed: [hostname]
Report URL: https://bencher.dev/console/projects/[project-name]/reports/[report-uuid]
```

The report URL provides direct access to the uploaded results in the Bencher web
interface.
