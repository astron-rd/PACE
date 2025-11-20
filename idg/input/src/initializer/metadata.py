import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True, cache=True)
def compute_metadata(
    grid_size: int,
    subgrid_size: int,
    nr_channels: int,
    bl: int,
    u_pixels: np.ndarray,
    v_pixels: np.ndarray,
    max_group_size: int,
) -> list:
    """
    Compute the metadata for a given baseline.

    :param grid_size: size of the grid
    :param subgrid_size: size of the subgrid
    :param nr_channels: number of frequency channels
    :param bl: baseline index
    :param u_pixels: U coordinates of the visibilities
    :param v_pixels: V coordinates of the visibilities
    :param max_group_size: maximum number of visibilities (timesteps) in a group
    :return: metadata array, shape (nr_subgrids)
    """
    metadata = []

    nr_timesteps = u_pixels.shape[0]
    max_distance = 0.8 * subgrid_size

    # Iterate all timesteps
    timestep = 0
    while timestep < nr_timesteps:
        # Start a new group
        current_u = u_pixels[timestep]
        current_v = v_pixels[timestep]

        # Find consecutive timesteps that are close enough
        group_size = 1
        while (
            timestep + group_size < nr_timesteps
            and group_size < max_group_size
            and np.sqrt(
                (u_pixels[timestep + group_size] - current_u) ** 2
                + (v_pixels[timestep + group_size] - current_v) ** 2
            )
            <= max_distance
        ):
            group_size += 1

        # Calculate group center and subgrid coordinate
        group_u = np.mean(u_pixels[timestep : timestep + group_size])
        group_v = np.mean(v_pixels[timestep : timestep + group_size])

        subgrid_x = int(group_u - (subgrid_size / 2))
        subgrid_y = int(group_v - (subgrid_size / 2))
        subgrid_x = max(0, min(grid_size - subgrid_size, subgrid_x))
        subgrid_y = max(0, min(grid_size - subgrid_size, subgrid_y))

        # Create metadata
        metadata.append(
            (
                bl,
                timestep,
                group_size,
                0,
                nr_channels,
                (subgrid_x, subgrid_y, 0),
            ),
        )
        timestep += group_size

    return metadata


def init_metadata(
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
            coordinate_type = np.dtype([("x", np.intc), ("y", np.intc), ("z", np.intc)])
            metadata_type = np.dtype(
                [
                    ("baseline", np.intc),
                    ("time_index", np.intc),
                    ("nr_timesteps", np.intc),
                    ("channel_begin", np.intc),
                    ("channel_end", np.intc),
                    ("coordinate", coordinate_type),
                ]
            )

            metadata.append(np.asarray(m, dtype=metadata_type))

    return np.asarray(metadata)
