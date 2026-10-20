import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def polyval(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Numba-compatible polynomial evaluation (equivalent to np.polyval)."""
    result = np.zeros_like(x)
    for i in range(len(x)):
        val = coefficients[0]
        for j in range(1, len(coefficients)):
            val = val * x[i] + coefficients[j]
        result[i] = val
    return result


@nb.njit(fastmath=True)
def evaluate_spheroidal(nu: np.ndarray) -> np.ndarray:
    """Evaluate the prolate spheroidal wave function."""
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

    result = np.zeros_like(nu)

    for part, end in [(0, 0.75), (1, 1.00)]:
        mask = (nu >= (0.0 if part == 0 else 0.75)) & (nu <= end)
        if not np.any(mask):
            continue

        nu_part = nu[mask]
        nusq = nu_part**2
        delnusq = nusq - end**2

        top = polyval(p[part][::-1], delnusq)
        bot = polyval(q[part][::-1], delnusq)

        valid = bot != 0
        result_part = np.zeros_like(nu_part)
        result_part[valid] = (1.0 - nusq[valid]) * (top[valid] / bot[valid])
        result[mask] = result_part

    return result


def get_taper(subgrid_size: int) -> np.ndarray:
    """Construct the subgrid taper by evaluating the prolate spheroidal wave
    function over a 1D grid and taking the outer product."""
    x = np.abs(np.linspace(-1, 1, num=subgrid_size, endpoint=True))
    x_spheroidal = evaluate_spheroidal(x)
    taper = x_spheroidal[np.newaxis, :] * x_spheroidal[:, np.newaxis]
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
    metadata,
    w_step,
    grid_size,
    image_size,
    wavenumbers,
    visibilities,
    uvw,
    taper,
    nr_correlations_in,
    subgrid_size,
    subgrid,
) -> None:
    """Grid visibilities onto a single subgrid."""
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

    u_offset = (x_coordinate + subgrid_size / 2 - grid_size / 2) * (
        2 * np.pi / image_size
    )
    v_offset = (y_coordinate + subgrid_size / 2 - grid_size / 2) * (
        2 * np.pi / image_size
    )
    w_offset = 2 * np.pi * w_offset_in_lambda

    for y in range(subgrid_size):
        for x in range(subgrid_size):
            l = compute_l(x, subgrid_size, image_size)
            m_val = compute_m(y, subgrid_size, image_size)
            n = compute_n(l, m_val)

            pixels = compute_pixels(
                nr_correlations_out,
                nr_timesteps,
                offset,
                uvw,
                bl,
                l,
                m_val,
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
    """Grid visibilities onto subgrids (parallel over subgrids)."""
    nr_subgrids = metadata.shape[0]

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
    """Compute the phasor used to shift a subgrid to its position in the grid."""
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
    """Add a subgrid to the grid."""
    m = metadata[s]
    grid_x = m["coordinate"]["x"]
    grid_y = m["coordinate"]["y"]

    if (
        grid_x >= 0
        and grid_x < grid_size - subgrid_size
        and grid_y >= 0
        and grid_y < grid_size - subgrid_size
    ):
        for y in range(subgrid_size):
            for x in range(subgrid_size):
                x_src = int((x + (subgrid_size / 2)) % subgrid_size)
                y_src = int((y + (subgrid_size / 2)) % subgrid_size)

                for p in range(nr_correlations):
                    grid[p, grid_y + y, grid_x + x] += np.complex64(
                        subgrids[s, p, y_src, x_src] * phasor[y, x]
                    )
