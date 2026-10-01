import numba as nb
import numpy as np


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
