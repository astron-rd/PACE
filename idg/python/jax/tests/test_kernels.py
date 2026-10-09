import numpy as np
import pytest
from idg_jax.kernels.spheroidal import evaluate_spheroidal
from idg_jax.taper import get_taper


def test_evaluate_spheroidal_returns_float():
    result = np.asarray(evaluate_spheroidal(np.array([0.5], dtype=np.float32)))
    assert result.dtype == np.float32


def test_evaluate_spheroidal_half():
    # Matches the numba reference: evaluate_spheroidal(0.5) = 0.27079904
    result = float(
        np.asarray(evaluate_spheroidal(np.array([0.5], dtype=np.float32)))[0]
    )
    assert result == pytest.approx(0.270799, abs=1e-5)


def test_taper_shape_and_finite():
    taper = get_taper(32)
    assert taper.shape == (32, 32)
    assert taper.dtype == np.float32
    assert np.all(np.isfinite(taper))


def test_taper_peak_at_center():
    # Matches the numba reference: center value is 0.989938, edges fall off
    taper = get_taper(32)
    center = taper[16, 16]
    assert center == pytest.approx(0.989938, abs=1e-4)
    assert taper[0, 0] < 0.5
