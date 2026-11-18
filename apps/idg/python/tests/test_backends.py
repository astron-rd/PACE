"""Tests for IDG Python kernels across backends"""

import numpy as np
import pytest

from idg.backends import available_backends, get_backend


@pytest.mark.parametrize("backend_name", available_backends())
def test_evaluate_spheroidal(backend_name, benchmark):
    backend = get_backend(backend_name)
    nu = np.linspace(0.0, 1.0, 1024)
    benchmark(backend.evaluate_spheroidal, nu)


def test_backends_produce_same_taper():
    tapers = {name: get_backend(name).get_taper(32) for name in available_backends()}
    reference = tapers[available_backends()[0]]
    for taper in tapers.values():
        assert taper.shape == (32, 32)
        np.testing.assert_allclose(taper, reference, atol=1e-6)


def test_backends_produce_same_phasor():
    phasors = {
        name: np.asarray(get_backend(name).compute_phasor(32))
        for name in available_backends()
    }
    reference = phasors[available_backends()[0]]
    for phasor in phasors.values():
        np.testing.assert_allclose(phasor, reference, atol=1e-6)


def _make_inputs(nr_baselines=2, nr_timesteps=8, nr_channels=4):
    """Small deterministic inputs, one subgrid per baseline with varying
    group sizes and the full channel range."""
    rng = np.random.default_rng(42)
    uvw = np.zeros(
        nr_baselines * nr_timesteps,
        dtype=[("u", np.float64), ("v", np.float64), ("w", np.float64)],
    )
    uvw["u"] = rng.normal(scale=0.1, size=nr_baselines * nr_timesteps)
    uvw["v"] = rng.normal(scale=0.1, size=nr_baselines * nr_timesteps)
    uvw["w"] = rng.normal(scale=0.01, size=nr_baselines * nr_timesteps)
    uvw = uvw.reshape(nr_baselines, nr_timesteps)

    visibilities = (
        rng.normal(size=(nr_baselines, nr_timesteps, nr_channels, 1))
        + 1j * rng.normal(size=(nr_baselines, nr_timesteps, nr_channels, 1))
    ).astype(np.complex64)

    coordinate_type = np.dtype([("x", np.uint32), ("y", np.uint32), ("z", np.uint32)])
    metadata_type = np.dtype(
        [
            ("baseline", np.uint32),
            ("time_index", np.uint32),
            ("nr_timesteps", np.uint32),
            ("channel_begin", np.uint32),
            ("channel_end", np.uint32),
            ("coordinate", coordinate_type),
        ]
    )
    groups = [(0, 3), (3, nr_timesteps - 3)]
    metadata = np.zeros(nr_baselines * len(groups), dtype=metadata_type)
    for b in range(nr_baselines):
        for g, (t0, nt) in enumerate(groups):
            m = metadata[b * len(groups) + g]
            m["baseline"] = b
            m["time_index"] = t0
            m["nr_timesteps"] = nt
            m["channel_begin"] = 0
            m["channel_end"] = nr_channels
            m["coordinate"] = (b, g, g)

    return uvw, visibilities, metadata


def _run_gridding(backend_name):
    """Grid a tiny input and add the subgrids to a grid with one backend."""
    backend = get_backend(backend_name)
    uvw, visibilities, metadata = _make_inputs()
    subgrid_size = 8
    grid_size = 32

    frequencies = np.linspace(1.0e8, 1.4e8, visibilities.shape[2])
    wavenumbers = (frequencies * 2 * np.pi) / 299792458.0
    image_size = 299792458.0 / frequencies[-1]

    taper = backend.get_taper(subgrid_size)
    subgrids = np.zeros(
        (metadata.shape[0], 1, subgrid_size, subgrid_size), dtype=np.complex64
    )
    backend.visibilities_to_subgrids(
        1.0,
        image_size,
        grid_size,
        wavenumbers,
        uvw,
        visibilities,
        taper,
        metadata,
        subgrids,
    )

    grid = np.zeros((1, grid_size, grid_size), dtype=np.complex64)
    phasor = np.asarray(backend.compute_phasor(subgrid_size))
    backend.add_subgrids_to_grid(
        metadata, subgrids, grid, phasor, 1, subgrid_size, grid_size
    )
    return grid


@pytest.mark.parametrize("backend_name", available_backends())
def test_grid_and_add_roundtrip(backend_name):
    grid = _run_gridding(backend_name)
    assert np.all(np.isfinite(grid.view(np.float32)))
    assert np.any(grid != 0)


def test_backends_produce_same_grid():
    grids = [_run_gridding(name) for name in available_backends()]
    np.testing.assert_allclose(grids[0], grids[-1], rtol=1e-5, atol=1e-6)
