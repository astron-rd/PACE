import numpy as np
from numba import jit, float32


@jit(nopython=True, fastmath=True, cache=True)
def polyval(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Numba-compatible polynomial evaluation (equivalent to np.polyval).

    :param coeffs: Polynomial coefficients in descending order [a_n, a_{n-1}, ..., a_0]
    :param x: Value(s) at which to evaluate the polynomial
    :return Array of evaluated polynomial values, shape (len(x))
    """
    result = np.zeros_like(x)
    for i in range(len(x)):
        val = coefficients[0]
        for j in range(1, len(coefficients)):
            val = val * x[i] + coefficients[j]
        result[i] = val
    return result


@jit(nopython=True, fastmath=True, cache=False)
def evaluate_spheroidal(nu: np.ndarray) -> np.ndarray:
    """
    Evaluate the prolate spheroidal wave function.

    param: nu: parameters of the spheroidal wave function
    return: array with value of the spheroidal wave function, shape (len(nu))
    """
    P = np.array(
        [
            [8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
            [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2],
        ]
    )
    Q = np.array(
        [
            [1.0000000e0, 8.212018e-1, 2.078043e-1],
            [1.0000000e0, 9.599102e-1, 2.918724e-1],
        ]
    )

    # Create result array
    result = np.zeros_like(nu)

    # Process each part
    for part, end in [(0, 0.75), (1, 1.00)]:
        mask = (nu >= (0.0 if part == 0 else 0.75)) & (nu <= end)
        if not np.any(mask):
            continue

        nu_part = nu[mask]
        nusq = nu_part**2
        delnusq = nusq - end**2

        # Calculate polynomial using Horner's method
        top = polyval(P[part][::-1], delnusq)
        bot = polyval(Q[part][::-1], delnusq)

        # Avoid division by zero
        valid = bot != 0
        result_part = np.zeros_like(nu_part)
        result_part[valid] = (1.0 - nusq[valid]) * (top[valid] / bot[valid])
        result[mask] = result_part

    return result


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def add_pt_src_to_baseline(
    bl: int,  # baseline index
    nr_time: int,  # number of timesteps
    nr_channels: int,  # number of frequency channels
    amplitude: complex,  # amplitude of the point source
    frequencies: np.ndarray,  # frequencies of the channels
    uvw: np.ndarray,  # uvw coordinates
    l: float,  # l coordinate of the point source
    m: float,  # m coordinate of the point source
    visibilities: np.ndarray,  # visibility data
) -> None:
    """
    Update the visibility data for a single baseline with a point source.

    :param bl: baseline index
    :param nr_time: number of timesteps
    :param nr_channels: number of frequency channels
    :param amplitude: amplitude of the point source
    :param frequencies: frequencies of the channels
    :param uvw: uvw coordinates
    :param l: l coordinate of the point source
    :param m: m coordinate of the point source
    :param visibilities: visibility array, shape (nr_baselines, nr_time, nr_channels, nr_correlations)
    """
    speed_of_light = 299792458.0

    for t in range(nr_time):
        for c in range(nr_channels):
            u = (frequencies[c] / speed_of_light) * uvw["u"][bl, t]
            v = (frequencies[c] / speed_of_light) * uvw["v"][bl, t]

            phase = -2 * np.pi * (u * l + v * m)
            value = amplitude * np.exp(1j * phase)

            visibilities[bl, t, c, :] += value


@jit(nopython=True, fastmath=False, cache=True, nogil=True)
def visibilities_to_subgrid(
    s: int,
    metadata: np.ndarray,
    w_step: float,
    grid_size: int,
    image_size: float,
    wavenumbers: np.ndarray,
    visibilities: np.ndarray,
    uvw: np.ndarray,
    taper: np.ndarray,
    nr_correlations_in: int,
    subgrid_size: int,
    subgrid: np.ndarray,
) -> None:
    """
    Grid visibilities onto a subgrid.

    :param s: subgrid index
    :param metadata: metadata for the subgrid
    :param w_step: w step in wavelengths
    :param grid_size: grid size in pixels
    :param image_size: image size in radians
    :param wavenumbers: wavenumbers of the frequencies
    :param visibilities: visibility data
    :param uvw: uvw coordinates
    :param taper: taper function
    :param nr_correlations_in: number of input correlations
    :param subgrid_size: subgrid size in pixels
    :param subgrid: subgrid array
    """
    # Load metadata
    m = metadata[s]
    bl = m["baseline"]
    offset = m["time_index"]
    nr_timesteps = m["nr_timesteps"]
    channel_begin = m["channel_begin"]
    channel_end = m["channel_end"]
    x_coordinate = m["coordinate"]["x"]
    y_coordinate = m["coordinate"]["y"]
    w_offset_in_lambda = w_step * (m["coordinate"]["z"] + 0.5)
    nr_correlations_out = 4 if nr_correlations_in == 4 else 1

    # Compute offsets
    u_offset = (x_coordinate + subgrid_size / 2 - grid_size / 2) * (
        2 * np.pi / image_size
    )
    v_offset = (y_coordinate + subgrid_size / 2 - grid_size / 2) * (
        2 * np.pi / image_size
    )
    w_offset = 2 * np.pi * w_offset_in_lambda

    # Precompute constants
    half_subgrid = subgrid_size / 2
    image_scale = image_size / subgrid_size

    for y in range(subgrid_size):
        for x in range(subgrid_size):

            # Compute l, m, n
            l = float32((x + 0.5 - half_subgrid) * image_scale)
            m = float32((y + 0.5 - half_subgrid) * image_scale)
            tmp = float32(l * l + m * m)  # type: ignore
            n = float32(tmp / (1.0 + np.sqrt(1.0 - tmp)))  # type: ignore

            # Compute pixels
            pixels = np.zeros(nr_correlations_out, dtype=np.complex64)

            for time in range(nr_timesteps):
                idx = offset + time
                u = uvw["u"][bl][idx]
                v = uvw["v"][bl][idx]
                w = uvw["w"][bl][idx]

                phase_index = float32(u * l + v * m + w * n)
                phase_offset = float32(u_offset * l + v_offset * m + w_offset * n)

                for chan in range(channel_begin, channel_end):
                    phase = float32(phase_offset - (phase_index * wavenumbers[chan]))
                    phasor = np.exp(1j * phase)  # type: ignore

                    for pol in range(nr_correlations_in):
                        pixels[pol % nr_correlations_out] += (
                            visibilities[bl, idx, chan, pol] * phasor
                        )

            # Apply taper and store
            sph = taper[y, x]
            x_dst = int((x + half_subgrid) % subgrid_size)
            y_dst = int((y + half_subgrid) % subgrid_size)

            for pol in range(nr_correlations_out):
                subgrid[pol, y_dst, x_dst] = pixels[pol] * sph


@jit(nopython=True, fastmath=True, cache=True)
def compute_phasor(subgrid_size: int) -> np.ndarray:
    """
    Compute the phasor which is used to shift the subgrid to the correct position
    in the grid.

    :param subgrid_size: size of the subgrid
    :return: phasor array, shape (subgrid_size, subgrid_size)
    """
    phasor = np.zeros(shape=(subgrid_size, subgrid_size), dtype=np.complex64)
    for y in range(subgrid_size):
        for x in range(subgrid_size):
            phase = np.float32(np.pi * (x + y - subgrid_size) / subgrid_size)
            phasor[y, x] = np.exp(1j * phase)
    return phasor


@jit(nopython=True, fastmath=True, cache=True)
def add_subgrid_to_grid(
    s: int,
    metadata: np.ndarray,
    subgrids: np.ndarray,
    grid: np.ndarray,
    phasor: np.ndarray,
    nr_correlations: int,
    subgrid_size: int,
    grid_size: int,
) -> None:
    """
    Add a subgrid to the grid.

    :param s: subgrid index
    :param metadata: metadata array
    :param subgrids: subgrid array
    :param grid: grid array
    :param phasor: phasor array
    :param nr_correlations: number of correlations
    :param subgrid_size: size of the subgrid
    :param grid_size: size of the grid
    """
    # Load metadata
    m = metadata[s]

    # Load position in grid
    coordinate = m["coordinate"]
    grid_x = coordinate["x"]
    grid_y = coordinate["y"]

    # Check whether subgrid fits in grid
    if (
        grid_x >= 0
        and grid_x < grid_size - subgrid_size
        and grid_y >= 0
        and grid_y < grid_size - subgrid_size
    ):
        for y in range(subgrid_size):
            for x in range(subgrid_size):
                # Compute shifted position in subgrid
                x_src = int((x + (subgrid_size / 2)) % subgrid_size)
                y_src = int((y + (subgrid_size / 2)) % subgrid_size)

                # Add subgrid value to grid
                for p in range(nr_correlations):
                    grid[p, grid_y + y, grid_x + x] += np.complex64(
                        subgrids[s, p, y_src, x_src] * phasor[y, x]
                    )


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
