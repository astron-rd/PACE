import random
import numpy as np
import numba as nb

import idgtypes

from kernels import add_pt_src_to_baseline, evaluate_spheroidal, compute_metadata


def get_uvw(
    observation_hours: float,
    nr_baselines: int,
    grid_size: int,
    ellipticity: float = 0.1,
    seed: int = 2,
) -> np.ndarray:
    """
    Generate simulated UVW data

     Parameters:
     :param grid_size: Size of the image in pixels (assumed square)
     :param observation_hours: Duration of observation in hours
     :param nr_baselines: Number of baselines to simulate
     :param ellipticity: Amount of ellipticity (0=circular, 1=highly elliptical)
     :param seed: Random seed for generating baseline ratios and starting angles

     :return uvw array, shape (nr_baselines, nr_timesteps)
    """

    # Convert observation time to seconds (1 sample per second)
    observation_seconds = int(observation_hours * 3600)
    time_samples = np.linspace(0, observation_seconds - 1, observation_seconds)
    nr_timesteps = observation_seconds

    # Initialize uvw array
    uvw = np.zeros(shape=(nr_baselines, nr_timesteps), dtype=idgtypes.uvwtype)

    # Calculate maximum UV distance
    max_u = 0.7 * (grid_size / 2)
    max_v = 0.7 * (grid_size / 2)

    # Generate baseline ratios with more short baselines (beta distribution)
    # Beta distribution with alpha=1, beta=3 peaks at 0 and decreases
    np.random.seed(seed)
    baseline_ratios = np.random.beta(1, 3, nr_baselines)

    # Generate random starting angles for each baseline
    start_angles = 2 * np.pi * np.random.rand(nr_baselines)

    # Calculate the UV coordinates for each baseline
    for bl, ratio in enumerate(baseline_ratios):
        # Calculate radius for this baseline
        u_radius = ratio * max_u
        v_radius = ratio * max_v

        # Apply ellipticity if specified
        if ellipticity > 0:
            # Make the ellipse orientation depend on the baseline
            # Longer baselines have more ellipticity
            ellipse_factor = 1.0 + ellipticity * ratio
            u_radius *= ellipse_factor
            v_radius /= ellipse_factor

        # Calculate angular velocity (complete circle in 24 hours)
        # For shorter observations, we get an arc instead of full circle
        angular_velocity = 2 * np.pi / (24 * 3600)  # rad/sec

        # Generate UV coordinates with random starting angle
        angle = start_angles[bl] + angular_velocity * time_samples
        u_coords = u_radius * np.cos(angle)
        v_coords = v_radius * np.sin(angle)

        # Store the coordinates
        uvw["u"][bl, :] = u_coords + grid_size / 2
        uvw["v"][bl, :] = v_coords + grid_size / 2

    return uvw


def get_frequencies(
    start_frequency: float, frequency_increment: float, nr_channels: int
) -> np.ndarray:
    """
    Generate array of frequencies for each channel.

    :param start_frequency: Starting frequency in Hz
    :param frequency_increment: Increment in Hz between consecutive channels
    :param nr_channels: Number of frequency channels

    :return frequencies array, shape (nr_channels)
    """
    return np.arange(
        start_frequency,
        start_frequency + nr_channels * frequency_increment,
        frequency_increment,
    )


def get_metadata(
    nr_channels: int,
    subgrid_size: int,
    grid_size: int,
    uvw: np.ndarray,
    max_group_size: int = 256,
) -> np.ndarray:
    """
    Compute metadata for all baselines.

    :param nr_channels: number of frequency channels
    :param subgrid_size: size of the subgrid
    :param grid_size: size of the grid
    :param uvw: array of uvw coordinates, shape (nr_baselines, nr_timesteps, 3)
    :param max_group_size: maximum number of visibilities (timesteps) in a group

    :return metadata array, shape (nr_subgrids)
    """

    # Get the U and V coordinates for each baseline
    u_pixels = uvw["u"]
    v_pixels = uvw["v"]

    # Get the number of baselines
    nr_baselines = uvw.shape[0]

    # Initialize the metadata list
    metadata = []

    # Compute the metadata for each baseline
    for bl in range(nr_baselines):
        for m in compute_metadata(
            grid_size,
            subgrid_size,
            nr_channels,
            bl,
            u_pixels[bl, :],
            v_pixels[bl, :],
            max_group_size,
        ):
            metadata.append(np.asarray(m, dtype=idgtypes.metadatatype))

    return np.asarray(metadata)


@nb.njit(cache=True, parallel=True)
def get_visibilities(
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
        dtype=idgtypes.visibilitiestype,
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
        l = offset[0] * image_size / grid_size
        m = offset[1] * image_size / grid_size

        for bl in nb.prange(nr_baselines):
            add_pt_src_to_baseline(
                bl,
                nr_timesteps,
                nr_channels,
                amplitude,
                frequencies,
                uvw,
                l,
                m,
                visibilities,
            )

    return visibilities


def get_taper(subgrid_size: int) -> np.ndarray:
    """
    Construct taper for subgrid

    The taper is constructed by evaluating the prolate spheroidal wave function
    at a set of points from -1 to 1 in the x and y directions. The result is a
    2D array that is used to weigh the subgrid pixels.

    :param subgrid_size: size of the subgrid
    :return: taper array, shape (subgrid_size, subgrid_size)
    """

    # Evaluate prolate spheroidal wave function
    x = np.abs(np.linspace(-1, 1, num=subgrid_size, endpoint=False))
    x_spheroidal = evaluate_spheroidal(x)

    # Construct 2D taper array
    taper = x_spheroidal[np.newaxis, :] * x_spheroidal[:, np.newaxis]

    # Cast to correct type
    return taper.astype(idgtypes.tapertype)
