import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def polyval(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Numba-compatible polynomial evaluation (equivalent to np.polyval).

    :param coefficients: Polynomial coefficients in descending order [a_n, a_{n-1}, ..., a_0]
    :param x: Value(s) at which to evaluate the polynomial
    :return: Array of evaluated polynomial values, shape (len(x))
    """
    result = np.zeros_like(x)
    for i in range(len(x)):
        val = coefficients[0]
        for j in range(1, len(coefficients)):
            val = val * x[i] + coefficients[j]
        result[i] = val
    return result


@nb.njit(fastmath=True)
def evaluate_spheroidal(nu: np.ndarray) -> np.ndarray:
    """
    Evaluate the prolate spheroidal wave function.

    :param nu: Parameters of the spheroidal wave function.
    :return: Array with values of the spheroidal wave function, shape (len(nu)).
    """
    p = np.array(
        [
            [8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
            [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2],
        ]
    )
    q = np.array(
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
        top = polyval(p[part][::-1], delnusq)
        bot = polyval(q[part][::-1], delnusq)

        # Avoid division by zero
        valid = bot != 0
        result_part = np.zeros_like(nu_part)
        result_part[valid] = (1.0 - nusq[valid]) * (top[valid] / bot[valid])
        result[mask] = result_part

    return result


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
    x = np.abs(np.linspace(-1, 1, num=subgrid_size, endpoint=True))
    x_spheroidal = evaluate_spheroidal(x)

    # Construct 2D taper array
    taper = x_spheroidal[np.newaxis, :] * x_spheroidal[:, np.newaxis]

    # Cast to correct type
    return taper.astype(np.float32)


@nb.njit(fastmath=True)
def compute_pixels(
    nr_correlations_out,
    nr_timesteps,
    offset,
    uvw,
    bl,
    l,
    m,
    n,
    u_offset,
    v_offset,
    w_offset,
    channel_begin,
    channel_end,
    wavenumbers,
    nr_correlations_in,
    visibilities,
):
    pixels = np.zeros(nr_correlations_out, dtype=np.complex64)

    for time in range(nr_timesteps):
        idx = offset + time
        u = uvw["u"][bl][idx]
        v = uvw["v"][bl][idx]
        w = uvw["w"][bl][idx]

        phase_index = nb.float32(u * l + v * m + w * n)
        phase_offset = nb.float32(u_offset * l + v_offset * m + w_offset * n)

        for chan in range(channel_begin, channel_end):
            phase = nb.float32(phase_offset - (phase_index * wavenumbers[chan]))
            phasor = np.exp(1j * phase)

            for pol in range(nr_correlations_in):
                pixels[pol % nr_correlations_out] += (
                    visibilities[bl, idx, chan, pol] * phasor
                )

    return pixels


@nb.njit
def compute_l(x: int, subgrid_size: int, image_size: float) -> float:
    return (x + 0.5 - (subgrid_size / 2.0)) * image_size / subgrid_size


@nb.njit
def compute_m(y: int, subgrid_size: int, image_size: float) -> float:
    return compute_l(y, subgrid_size, image_size)


@nb.njit
def compute_n(l: float, m: float) -> float:
    tmp = l * l + m * m

    if tmp >= 1.0:
        return 1.0

    return tmp / (1.0 + np.sqrt(1.0 - tmp))


@nb.njit(cache=True)
def visibilities_to_subgrid(
    metadata: dict,
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
    m = metadata
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

    for y in range(subgrid_size):
        for x in range(subgrid_size):
            # Compute l, m, n
            l = compute_l(x, subgrid_size, image_size)
            m = compute_m(y, subgrid_size, image_size)
            n = compute_n(l, m)

            # Compute pixels
            pixels = compute_pixels(
                nr_correlations_out,
                nr_timesteps,
                offset,
                uvw,
                bl,
                l,
                m,
                n,
                u_offset,
                v_offset,
                w_offset,
                channel_begin,
                channel_end,
                wavenumbers,
                nr_correlations_in,
                visibilities,
            )

            # Apply taper and store
            sph = taper[y, x]
            x_dst = int((x + (subgrid_size / 2)) % subgrid_size)
            y_dst = int((y + (subgrid_size / 2)) % subgrid_size)

            for pol in range(nr_correlations_out):
                subgrid[pol, y_dst, x_dst] = pixels[pol] * sph


@nb.njit(parallel=True)
def visibilities_to_subgrids(
    w_step,
    image_size,
    grid_size,
    wavenumbers,
    uvw,
    visibilities,
    taper,
    metadata,
    subgrids,
):
    """
    Grid visibilities onto subgrids.

    :param w_step: w step in wavelengths
    :param image_size: image size in radians
    :param grid_size: grid size in pixels
    :param wavenumbers: wavenumbers of the frequencies
    :param uvw: uvw coordinates
    :param visibilities: visibility data
    :param taper: taper function
    :param metadata: metadata array
    :param subgrids: subgrid array
    """
    nr_subgrids = metadata.shape[0]

    # Grid visibilities onto subgrids
    for s in nb.prange(nr_subgrids):
        visibilities_to_subgrid(
            metadata[s],
            w_step,
            grid_size,
            image_size,
            wavenumbers,
            visibilities,
            uvw,
            taper,
            visibilities.shape[3],
            subgrids.shape[2],
            subgrids[s],
        )
    return subgrids


@nb.njit(fastmath=True)
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


@nb.njit(fastmath=True)
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


@nb.njit
def add_subgrids_to_grid(
    metadata: np.ndarray,
    subgrids: np.ndarray,
    grid: np.ndarray,
    phasor: np.ndarray,
    nr_correlations: int,
    subgrid_size: int,
    grid_size: int,
) -> None:
    """
    Add subgrids to the grid.

    Runs over the subgrids serially: subgrid footprints overlap in the grid,
    so parallel accumulation would be a data race without per-pixel atomics.

    :param metadata: metadata array
    :param subgrids: subgrid array
    :param grid: grid array
    :param phasor: phasor array
    :param nr_correlations: number of correlations
    :param subgrid_size: size of the subgrid
    :param grid_size: size of the grid
    """
    for s in range(subgrids.shape[0]):
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
