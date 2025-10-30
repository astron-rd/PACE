from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import numpy as np

import idgtypes
from kernels import add_subgrid_to_grid, compute_phasor, visibilities_to_subgrid


class Gridder:

    def __init__(self, nr_correlations_in: int, subgrid_size: int):
        self.nr_correlations_in = nr_correlations_in
        self.nr_correlations_out = 4 if nr_correlations_in == 4 else 1
        self.subgrid_size = subgrid_size

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
        """
        Grid visibilities onto subgrids.

        :param w_step: w step in wavelengths
        :param image_size: image size in radians
        :param grid_size: size of the grid
        :param wavenumbers: array of wavenumbers, shape (nr_channels)
        :param uvw: array of uvw coordinates, shape (nr_baselines, nr_time, 3)
        :param visibilities: array of visibilities, shape (nr_baselines, nr_time, nr_channels, nr_correlations_in)
        :param taper: array of the taper function, shape (subgrid_size, subgrid_size)
        :param metadata: array with metadata, shape (nr_subgrids)
        :param subgrids: array of subgrids, shape (nr_subgrids, nr_correlations_out, subgrid_size, subgrid_size)
        """
        assert self.nr_correlations_in == visibilities.shape[3]
        assert self.nr_correlations_out == subgrids.shape[1]
        assert self.subgrid_size == subgrids.shape[2]
        nr_subgrids = metadata.shape[0]

        # Grid visibilities onto subgrids
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = []
            for s in range(nr_subgrids):
                future = executor.submit(
                    visibilities_to_subgrid,
                    s,
                    metadata,
                    w_step,
                    grid_size,
                    image_size,
                    wavenumbers,
                    visibilities,
                    uvw,
                    taper,
                    self.nr_correlations_in,
                    self.subgrid_size,
                    subgrids[s],
                )
                futures.append(future)

            for future in as_completed(futures):
                pass

        # Apply FFT to each subgrid
        subgrids[:] = np.fft.ifft2(subgrids, axes=(2, 3))

    def add_subgrids_to_grid(
        self,
        metadata: np.ndarray,  # metadata array
        subgrids: np.ndarray,  # subgrid array
        grid: np.ndarray,  # grid array
    ) -> None:
        """
        Add subgrids to the grid.

        :param metadata: metadata arraym shape (nr_subgrids)
        :param subgrids: subgrid array, shape (nr_subgrids, nr_correlations_out, subgrid_size, subgrid_size)
        :param grid: grid array (nr_correlations_out, grid_size, grid_size)
        """

        nr_correlations = grid.shape[0]
        grid_size = grid.shape[1]
        nr_subgrids = subgrids.shape[0]
        subgrid_size = subgrids.shape[2]

        # Compute phasor
        phasor = compute_phasor(subgrid_size)

        # Iterate all subgrids
        for s in range(nr_subgrids):
            add_subgrid_to_grid(
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
        """
        Transform Fourier Domain<->Image Domain.

        :param direction: idg.FourierDomainToImageDomain or idg.ImageDomainToFourierDomain
        :param grid array, shape (nr_correlations_out, grid_size, grid_size)
        """
        assert self.nr_correlations_out == grid.shape[0]
        height = grid.shape[1]
        width = grid.shape[2]
        assert height == width

        # FFT shift
        grid[:] = np.fft.fftshift(grid, axes=(1, 2))

        # FFT
        if direction == idgtypes.FOURIER_DOMAIN_TO_IMAGE_DOMAIN:
            grid[:] = np.fft.ifft2(grid, axes=(1, 2))
        else:
            grid[:] = np.fft.fft2(grid, axes=(1, 2))

        # FFT shift
        grid[:] = np.fft.fftshift(grid, axes=(1, 2))

        # Scaling
        scale = 2 + 0j
        if direction == idgtypes.FOURIER_DOMAIN_TO_IMAGE_DOMAIN:
            grid *= scale
        else:
            grid /= scale
