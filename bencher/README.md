# Bencher upload script

The `upload.py` script uploads benchmarking results to Bencher (https://bencher.dev/) for performance tracking and analysis.

## Prerequisites

Set the following environment variables:
```
export BENCHER_PROJECT="your-project-name"
export BENCHER_API_TOKEN="your-api-token"
```

## Usage

Upload benchmarking results from a JSON file:
```
python3 bencher_upload.py --project $BENCHER_PROJECT --token $BENCHER_API_TOKEN results.json --benchmark idg --branch python results.json
```

### Arguments
- `project`: Bencher UUID
- `token`: Bencher API token
- `benchmark`: The application under test, e.g., 'idg'
- `branch`: The specific implementation of this benchmark, e.g. 'python'. (Not the Git branch name!)
- `results.json`: Path to JSON file with benchmarking results

### Successful Output
Upon successful upload, you'll see:
```
Successfully uploaded X metrics to Bencher
Project: [project], Branch: [Git branch], Hash: [Git hash], Testbed: [hostname]
Report URL: https://bencher.dev/console/projects/[project-name]/reports/[report-uuid]
```

The report URL provides direct access to the uploaded results in the Bencher web interface.