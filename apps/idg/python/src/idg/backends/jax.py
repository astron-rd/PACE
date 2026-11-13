import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

# Number of subgrids processed in each jitted, vmapped batch. XLA parallelizes
# the batch axis (vectorize-over-an-axis). Chunking keeps the
# (batch, nt, nch, sg, sg) intermediate bounded; on an exclusive 72-core node
# the total time is nearly flat across 16..256 (70-77 s on the 10830-subgrid
# input), 256 is marginally fastest, so it stays the default.
_BATCH_SIZE = 256


@jax.jit
def _polyval(coefficients, x):
    """
    JAX-compatible polynomial evaluation (equivalent to np.polyval).

    :param coefficients: Polynomial coefficients in descending order [a_n, a_{n-1}, ..., a_0]
    :param x: Value(s) at which to evaluate the polynomial
    :return: Array of evaluated polynomial values, shape (len(x))
    """
    result = jnp.zeros_like(x)
    for c in coefficients:
        result = result * x + c
    return result


@jax.jit
def evaluate_spheroidal(nu):
    """
    Evaluate the prolate spheroidal wave function.

    :param nu: Parameters of the spheroidal wave function.
    :return: Array with values of the spheroidal wave function, shape (len(nu)).
    """
    p = jnp.array(
        [
            [8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
            [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2],
        ]
    )
    q = jnp.array(
        [
            [1.0000000e0, 8.212018e-1, 2.078043e-1],
            [1.0000000e0, 9.599102e-1, 2.918724e-1],
        ]
    )

    result = jnp.zeros_like(nu)

    # Process each part
    for part, end in [(0, 0.75), (1, 1.00)]:
        lo = 0.0 if part == 0 else 0.75
        mask = (nu >= lo) & (nu <= end)
        nusq = nu**2
        delnusq = nusq - end**2

        # Calculate polynomial using Horner's method
        top = _polyval(p[part][::-1], delnusq)
        bot = _polyval(q[part][::-1], delnusq)

        # Avoid division by zero
        valid = bot != 0
        term = jnp.where(valid, (1.0 - nusq) * (top / bot), 0.0)
        result = jnp.where(mask, term, result)

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
    x_spheroidal = np.asarray(evaluate_spheroidal(jnp.asarray(x)))

    # Construct 2D taper array
    taper = x_spheroidal[np.newaxis, :] * x_spheroidal[:, np.newaxis]

    # Cast to correct type
    return taper.astype(np.float32)


def compute_n(l, m):
    """Compute direction cosine n from l and m on the unit sphere."""
    tmp = l * l + m * m
    return jnp.where(tmp >= 1.0, 1.0, tmp / (1.0 + jnp.sqrt(1.0 - tmp)))


@jax.jit(static_argnames=("subgrid_size", "nr_corr_out"))
def _one_subgrid(
    u_s,
    v_s,
    w_s,
    vis_s,
    mask_s,
    wavenumbers,
    image_size,
    grid_size,
    subgrid_size,
    xc,
    yc,
    zc,
    w_step,
    nr_corr_out,
    taper,
):
    """
    Grid one subgrid's visibilities into a (nr_corr_out, sg, sg) array.

    Array-notation equivalent of the numba visibilities_to_subgrid kernel:
    the per-pixel phase is written as an einsum over timesteps and channels
    instead of nested loops, which XLA can vectorize. A lax.scan over
    timesteps was measured about 10x slower, so this form is kept.

    :param u_s: (nt,) per-timestep u for the subgrid's baseline
    :param v_s: (nt,) per-timestep v for the subgrid's baseline
    :param w_s: (nt,) per-timestep w for the subgrid's baseline
    :param vis_s: (nt, nch, nr_corr_in) visibilities, already sliced to the channel range
    :param mask_s: (nt,) 1 for real timesteps, 0 for padding
    :param wavenumbers: (nch,) wavenumbers of the frequencies
    :param image_size: image size in radians
    :param grid_size: size of the grid
    :param subgrid_size: size of the subgrid
    :param xc: subgrid x coordinate in the grid
    :param yc: subgrid y coordinate in the grid
    :param zc: subgrid w coordinate
    :param w_step: w step in wavelengths
    :param nr_corr_out: number of output correlations
    :param taper: (sg, sg) taper array
    """
    # Compute l, m, n (l varies with x, m varies with y)
    idx = jnp.arange(subgrid_size)
    c = (idx + 0.5 - subgrid_size / 2.0) * image_size / subgrid_size
    l = c[None, :]
    m = c[:, None]
    l_2d = jnp.broadcast_to(l, (subgrid_size, subgrid_size))
    m_2d = jnp.broadcast_to(m, (subgrid_size, subgrid_size))
    n = compute_n(l_2d, m_2d)

    # Compute offsets
    u_offset = (xc + subgrid_size / 2 - grid_size / 2) * (2 * np.pi / image_size)
    v_offset = (yc + subgrid_size / 2 - grid_size / 2) * (2 * np.pi / image_size)
    w_offset = 2 * np.pi * w_step * (zc + 0.5)
    phase_offset = u_offset * l_2d + v_offset * m_2d + w_offset * n

    # Phase per timestep: (nt, sg, sg)
    phase_index = (
        u_s[:, None, None] * l_2d[None]
        + v_s[:, None, None] * m_2d[None]
        + w_s[:, None, None] * n[None]
    )
    wn = wavenumbers[None, :, None, None]
    phase = phase_offset[None, None] - phase_index[:, None] * wn

    phasor = jnp.exp(1j * phase)

    # Zero out padded timesteps so they contribute nothing
    phasor = phasor * mask_s[:, None, None, None]

    # Compute pixels
    nr_corr_in = vis_s.shape[2]
    total = jnp.einsum("tcp,tcxy->pxy", vis_s, phasor)

    # Map input correlations onto output correlations (pol % nr_corr_out)
    idx_pol = jnp.arange(nr_corr_in) % nr_corr_out
    out = jnp.zeros((nr_corr_out, subgrid_size, subgrid_size), dtype=jnp.complex64)
    out = out.at[idx_pol].add(total)

    # Apply taper and store at the fft-shifted position
    return jnp.roll(out * taper[None, :, :], subgrid_size // 2, axis=(1, 2))


def _one_subgrid_batched(
    u_b,
    v_b,
    w_b,
    vis_b,
    mask_b,
    wavenumbers,
    image_size,
    grid_size,
    subgrid_size,
    xc_b,
    yc_b,
    zc_b,
    w_step,
    nr_corr_out,
    taper,
):
    """
    vmap ``_one_subgrid`` across a batch of subgrids.

    The leading (batch) axis is vectorized, letting XLA parallelize across
    CPU cores. A batch of size one is valid.
    """
    return jax.vmap(
        _one_subgrid,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None, 0, 0, 0, None, None, None),
    )(
        u_b,
        v_b,
        w_b,
        vis_b,
        mask_b,
        wavenumbers,
        image_size,
        grid_size,
        subgrid_size,
        xc_b,
        yc_b,
        zc_b,
        w_step,
        nr_corr_out,
        taper,
    )


@jax.jit(static_argnames=("subgrid_size",))
def _add_subgrids_scan(
    grid,
    subs,
    phasor,
    x_coords,
    y_coords,
    subgrid_size,
):
    """
    Add many subgrids to one grid in a single jitted scan.

    The grid stays in XLA buffers for the whole scan (CPU-only jax install);
    only the final result is copied back to the NumPy grid. On the real
    10830-subgrid input this runs in tens of milliseconds instead of the
    per-subgrid host loop round trip (seconds). A batch of size one is valid.

    :param grid: (G, G) grid for one correlation
    :param subs: (n_valid, sg, sg) subgrids to add
    :param phasor: (sg, sg) phasor array
    :param x_coords: (n_valid,) x coordinates in the grid
    :param y_coords: (n_valid,) y coordinates in the grid
    :param subgrid_size: size of the subgrid
    """

    def step(grid, i):
        sub = subs[i]
        grid_x = x_coords[i]
        grid_y = y_coords[i]
        # Compute shifted position in subgrid
        sub_shift = jnp.roll(
            jnp.roll(sub, subgrid_size // 2, axis=0), subgrid_size // 2, axis=1
        )
        block = lax.dynamic_slice(grid, (grid_y, grid_x), (subgrid_size, subgrid_size))
        # Add subgrid value to grid
        updated = block + sub_shift * phasor
        grid = lax.dynamic_update_slice(grid, updated, (grid_y, grid_x))
        return grid, None

    grid, _ = lax.scan(step, grid, jnp.arange(subs.shape[0]))
    return grid


@jax.jit(static_argnames=("subgrid_size",))
def compute_phasor(subgrid_size: int):
    """
    Compute the phasor which is used to shift the subgrid to the correct position
    in the grid.

    :param subgrid_size: size of the subgrid
    :return: phasor array, shape (subgrid_size, subgrid_size)
    """
    y = jnp.arange(subgrid_size)
    x = jnp.arange(subgrid_size)
    phase = jnp.pi * (x[None, :] + y[:, None] - subgrid_size) / subgrid_size
    return jnp.exp(1j * phase)


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

    Subgrids are gridded in batches of ``_BATCH_SIZE``, each batch one
    vmapped jit call. Timesteps are padded to the batch maximum and masked
    so padded entries contribute nothing, because subgrids have varying
    numbers of timesteps.

    :param w_step: w step in wavelengths
    :param image_size: image size in radians
    :param grid_size: grid size in pixels
    :param wavenumbers: wavenumbers of the frequencies
    :param uvw: structured (u,v,w) arrays, shape (nr_baselines, nr_time)
    :param visibilities: visibility data, shape (nr_baselines, nr_time, nr_channels, nr_correlations_in)
    :param taper: taper function, shape (subgrid_size, subgrid_size)
    :param metadata: metadata array
    :param subgrids: subgrid array (written in place)
    """
    u = uvw["u"]
    v = uvw["v"]
    w = uvw["w"]
    nr_corr_out = subgrids.shape[1]
    subgrid_size = subgrids.shape[2]
    nr_corr_in = visibilities.shape[3]
    # All subgrids cover the full channel range, so one slice suffices
    wn = np.asarray(wavenumbers, dtype=np.float64)[
        int(metadata["channel_begin"][0]) : int(metadata["channel_end"][0])
    ]
    taper_j = jnp.asarray(taper)

    nr_sg = int(np.max(metadata["nr_timesteps"]))
    nr_batches = subgrids.shape[0]
    nch = visibilities.shape[2]

    for start in range(0, nr_batches, _BATCH_SIZE):
        stop = min(start + _BATCH_SIZE, nr_batches)

        # Gather the batch into padded arrays
        u_b = np.zeros((stop - start, nr_sg), dtype=np.float64)
        v_b = np.zeros((stop - start, nr_sg), dtype=np.float64)
        w_b = np.zeros((stop - start, nr_sg), dtype=np.float64)
        vis_b = np.zeros((stop - start, nr_sg, nch, nr_corr_in), dtype=np.complex64)
        mask_b = np.zeros((stop - start, nr_sg), dtype=np.float32)
        xc = np.zeros(stop - start, dtype=np.int32)
        yc = np.zeros(stop - start, dtype=np.int32)
        zc = np.zeros(stop - start, dtype=np.int32)

        for i, s in enumerate(range(start, stop)):
            m = metadata[s]
            bl = int(m["baseline"])
            t0 = int(m["time_index"])
            nt = int(m["nr_timesteps"])
            cb = int(m["channel_begin"])
            ce = int(m["channel_end"])
            u_b[i, :nt] = u[bl, t0 : t0 + nt]
            v_b[i, :nt] = v[bl, t0 : t0 + nt]
            w_b[i, :nt] = w[bl, t0 : t0 + nt]
            vis_b[i, :nt] = visibilities[bl, t0 : t0 + nt, cb:ce, :]
            mask_b[i, :nt] = 1.0
            xc[i] = int(m["coordinate"]["x"])
            yc[i] = int(m["coordinate"]["y"])
            zc[i] = int(m["coordinate"]["z"])

        out = _one_subgrid_batched(
            u_b,
            v_b,
            w_b,
            vis_b,
            mask_b,
            wn,
            image_size,
            grid_size,
            subgrid_size,
            xc,
            yc,
            zc,
            w_step,
            nr_corr_out,
            taper_j,
        )
        subgrids[start:stop] = np.asarray(out)


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
    Add subgrids to the grid in one jitted scan per correlation.

    Subgrids whose coordinate lies outside the grid (the first/last row or
    column) are dropped on the host first, matching the guard in the numba
    kernel.

    :param metadata: metadata array
    :param subgrids: subgrid array
    :param grid: grid array
    :param phasor: phasor array
    :param nr_correlations: number of correlations
    :param subgrid_size: size of the subgrid
    :param grid_size: size of the grid
    """
    xs = np.asarray(metadata["coordinate"]["x"], dtype=np.int32)
    ys = np.asarray(metadata["coordinate"]["y"], dtype=np.int32)
    in_bounds = (
        (xs >= 0)
        & (xs < grid_size - subgrid_size)
        & (ys >= 0)
        & (ys < grid_size - subgrid_size)
    )
    idx = np.where(in_bounds)[0]
    if idx.size == 0:
        return
    for p in range(nr_correlations):
        grid[p] = np.asarray(
            _add_subgrids_scan(
                jnp.asarray(grid[p]),
                jnp.asarray(subgrids[idx, p]),
                jnp.asarray(phasor),
                xs[idx],
                ys[idx],
                subgrid_size,
            )
        )
