import numpy as np

from .backends import get_backend
from .types import FOURIER_DOMAIN_TO_IMAGE_DOMAIN


class Gridder:
    def __init__(self, backend, nr_correlations_in: int, subgrid_size: int):
        self.backend = backend
        self.nr_correlations_in = nr_correlations_in
        self.nr_correlations_out = 4 if nr_correlations_in == 4 else 1
        self.subgrid_size = subgrid_size

    @classmethod
    def create(cls, backend_name: str, nr_correlations_in: int, subgrid_size: int):
        """Build a Gridder for the named backend (``"numba"`` or ``"jax"``)."""
        return cls(get_backend(backend_name), nr_correlations_in, subgrid_size)

    def grid_onto_subgrids(
        self,
        w_step: float,
        image_size: float,
        grid_size: int,
        wavenumbers: np.ndarray,
        uvw: np.ndarray,
        visibilities: np.ndarray,
        taper: np.ndarray,
        metadata: np.ndarray,
        subgrids: np.ndarray,
    ) -> None:
        """Grid visibilities onto subgrids, then FFT each subgrid."""
        assert self.nr_correlations_in == visibilities.shape[3]
        assert self.nr_correlations_out == subgrids.shape[1]
        assert self.subgrid_size == subgrids.shape[2]

        self.backend.visibilities_to_subgrids(
            w_step,
            image_size,
            grid_size,
            wavenumbers,
            uvw,
            visibilities,
            taper,
            metadata,
            subgrids,
        )

        subgrids[:] = np.fft.ifft2(subgrids, axes=(2, 3))

    def add_subgrids_to_grid(
        self,
        metadata: np.ndarray,
        subgrids: np.ndarray,
        grid: np.ndarray,
    ) -> None:
        """Add subgrids to the grid."""
        nr_correlations = grid.shape[0]
        grid_size = grid.shape[1]
        nr_subgrids = subgrids.shape[0]
        subgrid_size = subgrids.shape[2]

        phasor = self.backend.compute_phasor(subgrid_size)

        for s in range(nr_subgrids):
            self.backend.add_subgrid_to_grid(
                s,
                metadata,
                subgrids,
                grid,
                phasor,
                nr_correlations,
                subgrid_size,
                grid_size,
            )

    def transform(self, direction: int, grid: np.ndarray) -> None:
        """Transform Fourier Domain <-> Image Domain."""
        assert self.nr_correlations_out == grid.shape[0]
        height = grid.shape[1]
        width = grid.shape[2]
        assert height == width

        grid[:] = np.fft.fftshift(grid, axes=(1, 2))

        if direction == FOURIER_DOMAIN_TO_IMAGE_DOMAIN:
            grid[:] = np.fft.ifft2(grid, axes=(1, 2))
        else:
            grid[:] = np.fft.fft2(grid, axes=(1, 2))

        grid[:] = np.fft.fftshift(grid, axes=(1, 2))

        scale = 2 + 0j
        if direction == FOURIER_DOMAIN_TO_IMAGE_DOMAIN:
            grid *= scale
        else:
            grid /= scale
