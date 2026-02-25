import numpy as np


def init_uvw(
    observation_hours: int,
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
    observation_seconds = observation_hours * 3600
    time_samples = np.arange(observation_seconds)
    nr_timesteps = observation_seconds

    # Initialize uvw array
    uvw_type = np.dtype([("u", np.float32), ("v", np.float32), ("w", np.float32)])
    uvw = np.zeros(shape=(nr_baselines, nr_timesteps), dtype=uvw_type)

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
