"""Benchmark tests for IDG Python kernels"""

import numpy as np

from idg_python.kernels.numba import evaluate_spheroidal


def test_evaluate_spheroidal(benchmark):
    nu = np.linspace(0.0, 1.0, 1024)
    benchmark.pedantic(evaluate_spheroidal, args=(nu,), warmup_rounds=1)
