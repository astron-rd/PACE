"""Benchmark tests for IDG Python kernels"""

import numpy as np
import pytest

from idg.backends import available_backends, get_backend


@pytest.mark.parametrize("backend_name", available_backends())
def test_evaluate_spheroidal(backend_name, benchmark):
    backend = get_backend(backend_name)
    nu = np.linspace(0.0, 1.0, 1024)
    benchmark(backend.evaluate_spheroidal, nu)
