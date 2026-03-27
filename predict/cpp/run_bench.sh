#!/bin/bash

# Check if the script is being run with two or three arguments
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <architecture> <compiler_version> [--bench-only-sources]"
    exit 1
fi

module load python

ARCHITECTURE=$1
COMPILER_VERSION=$2
FILTER=""

# Check if the --bench-only-sources flag is set. If so, append the Google benchmark filter
if [ "$3" == "--bench-only-sources" ]; then
    BENCH_ONLY_SOURCES=true
    FILTER="PredictBenchmark/PointSource PredictBenchmark/GaussianSource"
else
    BENCH_ONLY_SOURCES=false
fi

echo "Running on compiler version: ${COMPILER_VERSION}, architecture: ${ARCHITECTURE}, hostname: $(hostname), filter: ${FILTER}"

sbatch --wait -C ${ARCHITECTURE} -o output.txt -e error.txt ci/das6/compile_and_run.sh ${ARCHITECTURE} ${COMPILER_VERSION} ${FILTER}
cat output.txt >&1
cat error.txt >&2

python3 ci/summarize-results.py --filter PredictBenchmark/PointSource results*.json result-summary-pointsource
python3 ci/summarize-results.py --filter PredictBenchmark/GaussianSource results*.json result-summary-gaussian

if [ "$BENCH_ONLY_SOURCES" == false ]; then
    python3 ci/summarize-results.py --filter DirectionsBenchmark/Directions results*.json result-summary-directions
    python3 ci/summarize-results.py --filter PhasesBenchmark/ComputePhases results*.json result-summary-phases
    python3 ci/summarize-results.py --filter SpectrumBenchmark/Spectrum results*.json result-summary-spectrum
    python3 ci/summarize-results.py --filter SmearTermsBenchmark/SmearTermBenchmark results*.json result-summary-smearterms
fi