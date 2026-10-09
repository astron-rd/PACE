import numpy as np
import pytest
from idg_python.backends import available_backends, get_backend


@pytest.mark.parametrize("backend_name", available_backends())
def test_evaluate_spheroidal(backend_name):
    backend = get_backend(backend_name)
    nu = np.linspace(0.0, 1.0, 1024)
    result = np.asarray(backend.evaluate_spheroidal(nu))
    assert result.shape == nu.shape
    assert np.all(np.isfinite(result))


def test_backends_produce_same_taper():
    tapers = {}
    for name in available_backends():
        backend = get_backend(name)
        tapers[name] = backend.get_taper(32)
    a, b = tapers.values()
    assert a.shape == b.shape == (32, 32)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_backends_produce_same_phasor():
    phasors = {}
    for name in available_backends():
        backend = get_backend(name)
        phasors[name] = np.asarray(backend.compute_phasor(32))
    a, b = phasors.values()
    np.testing.assert_allclose(a, b, atol=1e-6)
