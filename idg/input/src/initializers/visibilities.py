import random
import numpy as np
import numba as nb


@nb.njit(fastmath=True, parallel=True)
def add_pt_src_to_baseline(
    baseline_index: int,
    nr_time: int,  # number of timesteps
    nr_channels: int,  # number of frequency channels
    amplitude: complex,  # amplitude of the point source
    frequencies: np.ndarray,  # frequencies of the channels
    uvw: np.ndarray,  # uvw coordinates
    l_coordinate: float,  # l coordinate of the point source
    m_coordinate: float,  # m coordinate of the point source
    visibilities: np.ndarray,  # visibility data
) -> None:
    """
    Update the visibility data for a single baseline with a point source.

    :param baseline_index: baseline index
    :param nr_time: number of timesteps
    :param nr_channels: number of frequency channels
    :param amplitude: amplitude of the point source
    :param frequencies: frequencies of the channels
    :param uvw: uvw coordinates
    :param l_coordinate: l coordinate of the point source
    :param m_coordinate: m coordinate of the point source
    :param visibilities: visibility array, shape (nr_baselines, nr_time, nr_channels, nr_correlations)
    """
    speed_of_light = 299792458.0

    for t in range(nr_time):
        for c in range(nr_channels):
            u = (frequencies[c] / speed_of_light) * uvw["u"][baseline_index, t]
            v = (frequencies[c] / speed_of_light) * uvw["v"][baseline_index, t]

            phase = -2 * np.pi * (u * l_coordinate + v * m_coordinate)
            value = amplitude * np.exp(1j * phase)

            visibilities[baseline_index, t, c, :] += value


@nb.njit(cache=True, parallel=True)
def init_visibilities(
    nr_correlations: int,
    nr_channels: int,
    nr_timesteps: int,
    nr_baselines: int,
    image_size: float,
    grid_size: int,
    frequencies: np.ndarray,
    uvw: np.ndarray,
    nr_point_sources: int = 4,
    max_pixel_offset: int = -1,
    random_seed: int = 2,
) -> np.ndarray:
    """
    Generate visibilities for given baseline and channel properties.

    :param nr_correlations: Number of correlations
    :param nr_channels: Number of frequency channels
    :param nr_timesteps: Number of timesteps
    :param nr_baselines: Number of baselines
    :param image_size: Size of the image in radians
    :param grid_size: Size of the grid in pixels
    :param frequencies: array of frequencies, shape (nr_channels)
    :param uvw: array of uvw coordinates, shape (nr_baselines, nr_timesteps, 3)
    :param nr_point_sources: Number of point sources to generate visibilities for
    :param max_pixel_offset: Maximum offset in pixels of the point sources from the center of the grid.
    :param random_seed: int, Random seed for generating the point sources

    :return: visibilities array, shape (nr_baselines, nr_time, nr_channels, nr_correlations)
    """

    if max_pixel_offset == -1:
        max_pixel_offset = grid_size // 3

    # Initialize visibilities to zero
    visibilities = np.zeros(
        (nr_baselines, nr_timesteps, nr_channels, nr_correlations),
        dtype=np.complex64,
    )

    # Create offsets for fake point sources
    offsets = list()
    random.seed(random_seed)
    for _ in range(nr_point_sources):
        x = (random.random() * (max_pixel_offset)) - (max_pixel_offset / 2)
        y = (random.random() * (max_pixel_offset)) - (max_pixel_offset / 2)
        offsets.append((x, y))

    # Update visibilities
    for offset in offsets:
        amplitude = 1

        # Convert offset from grid cells to radians (l,m coordinates)
        l_coordinate = offset[0] * image_size / grid_size
        m_coordinate = offset[1] * image_size / grid_size

        for bl in nb.prange(nr_baselines):
            add_pt_src_to_baseline(
                bl,
                nr_timesteps,
                nr_channels,
                amplitude,
                frequencies,
                uvw,
                l_coordinate,
                m_coordinate,
                visibilities,
            )

    return visibilities
