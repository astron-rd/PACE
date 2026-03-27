#!/bin/bash

set -e
# compile the code and run it on das6
# Specify the compiler version and architecture

ARCHITECTURE=$1
COMPILER_VERSION=$2

echo "Running on compiler version: ${COMPILER_VERSION}, architecture: ${ARCHITECTURE}, hostname: $(hostname)"

# A third argument may be passed. This is the benchmark filter
# Check if the filter args is available
if [ "$#" -ge 3 ]; then
    echo "Filter: $3"
    FILTER=$3
else
    FILTER=""
fi

BUILD_DIR=build-${COMPILER_VERSION}-${ARCHITECTURE}
module purge
module load spack/20250523
module load casacore
module load python
module load py-pip
module load boost
module load openblas
module load everybeam/0.7.1
module load cfitsio
module load hdf5

cmake -B ${BUILD_DIR} . -DCMAKE_BUILD_TYPE=Release -DPREDICT_BUILD_BENCHMARK=ON
make -C ${BUILD_DIR} -j

${BUILD_DIR}/benchmark/microbenchmarks --benchmark_filter=${FILTER} --benchmark_out=results-${COMPILER_VERSION}-${ARCHITECTURE}.json --benchmark_out_format=json --benchmark_min_warmup_time=1