import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


def compute_n(l, m):
    tmp = l * l + m * m
    return jnp.where(tmp >= 1.0, 1.0, tmp / (1.0 + jnp.sqrt(1.0 - tmp)))


@jax.jit(static_argnames=("subgrid_size", "nr_corr_out"))
def _one_subgrid(
    u_s,
    v_s,
    w_s,
    vis_s,
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
    """Grid one subgrid's visibilities into a (nr_corr_out, sg, sg) array.

    u_s/v_s/w_s: (nt,) per-timestep uvw for the subgrid's baseline.
    vis_s: (nt, nch, nr_corr_in) visibilities (already sliced cb:ce).
    wavenumbers: (nch,) (already sliced).
    taper: (sg, sg).
    """
    idx = jnp.arange(subgrid_size)
    c = (idx + 0.5 - subgrid_size / 2.0) * image_size / subgrid_size
    l = c[None, :]  # (1, sg)  l varies with x (column), matches compute_l(x)
    m = c[:, None]  # (sg, 1)  m varies with y (row), matches compute_m(y)
    l_2d = jnp.broadcast_to(l, (subgrid_size, subgrid_size))
    m_2d = jnp.broadcast_to(m, (subgrid_size, subgrid_size))
    n = compute_n(l_2d, m_2d)  # (sg, sg)

    u_offset = (xc + subgrid_size / 2 - grid_size / 2) * (2 * np.pi / image_size)
    v_offset = (yc + subgrid_size / 2 - grid_size / 2) * (2 * np.pi / image_size)
    w_offset = 2 * np.pi * w_step * (zc + 0.5)
    phase_offset = u_offset * l_2d + v_offset * m_2d + w_offset * n  # (sg, sg)

    # phase_index per timestep: (nt, sg, sg)
    phase_index = (
        u_s[:, None, None] * l_2d[None]
        + v_s[:, None, None] * m_2d[None]
        + w_s[:, None, None] * n[None]
    )
    wn = wavenumbers[None, :, None, None]  # (1, nch, 1, 1)
    phase = phase_offset[None, None] - phase_index[:, None, :, :] * wn  # (nt,nch,sg,sg)
    phasor = jnp.exp(1j * phase)  # (nt, nch, sg, sg)

    nr_corr_in = vis_s.shape[2]
    total = jnp.einsum("tcp,tcxy->pxy", vis_s, phasor)  # (nr_corr_in, sg, sg)

    # map input correlations onto output correlations (pol % nr_corr_out)
    idx_pol = jnp.arange(nr_corr_in) % nr_corr_out
    out = jnp.zeros((nr_corr_out, subgrid_size, subgrid_size), dtype=jnp.complex64)
    out = out.at[idx_pol].add(total)

    # taper and fft-shift (mirror numba: write pixels(y,x)*taper(y,x) into
    # position (y+sg/2, x+sg/2))
    out = out * taper[None, :, :]
    out = jnp.roll(out, subgrid_size // 2, axis=(1, 2))
    return out


@jax.jit(static_argnames=("subgrid_size",))
def _add_one_subgrid(grid, sub, phasor, grid_x, grid_y, subgrid_size):
    sub_shift = jnp.roll(
        jnp.roll(sub, subgrid_size // 2, axis=0), subgrid_size // 2, axis=1
    )
    block = lax.dynamic_slice(grid, (grid_y, grid_x), (subgrid_size, subgrid_size))
    updated = block + sub_shift * phasor
    grid = lax.dynamic_update_slice(grid, updated, (grid_y, grid_x))
    return grid


@jax.jit(static_argnames=("subgrid_size",))
def compute_phasor(subgrid_size: int):
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
    """Grid all subgrids. uvw: structured (u,v,w) arrays shape (nb, nt).
    visibilities: (nb, nt, nch, ncorr). metadata: structured array.
    taper: (sg, sg)."""
    u = uvw["u"]
    v = uvw["v"]
    w = uvw["w"]
    nr_corr_out = subgrids.shape[1]
    subgrid_size = subgrids.shape[2]
    wn = np.asarray(wavenumbers, dtype=np.float64)
    taper_j = jnp.asarray(taper)

    for s in range(subgrids.shape[0]):
        m = metadata[s]
        bl = int(m["baseline"])
        t0 = int(m["time_index"])
        nt = int(m["nr_timesteps"])
        cb = int(m["channel_begin"])
        ce = int(m["channel_end"])
        xc = int(m["coordinate"]["x"])
        yc = int(m["coordinate"]["y"])
        zc = int(m["coordinate"]["z"])

        u_s = np.asarray(u[bl, t0 : t0 + nt]).astype(np.float64)
        v_s = np.asarray(v[bl, t0 : t0 + nt]).astype(np.float64)
        w_s = np.asarray(w[bl, t0 : t0 + nt]).astype(np.float64)
        vis_s = np.asarray(visibilities[bl, t0 : t0 + nt, cb:ce, :]).astype(
            np.complex64
        )

        sub = _one_subgrid(
            u_s,
            v_s,
            w_s,
            vis_s,
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
        subgrids[s] = np.asarray(sub)


def add_subgrid_to_grid(
    s: int,
    metadata,
    subgrids,
    grid,
    phasor,
    nr_correlations,
    subgrid_size,
    grid_size,
):
    """Add subgrid ``s`` to the grid (matches the numba per-index interface)."""
    m = metadata[s]
    grid_x = int(m["coordinate"]["x"])
    grid_y = int(m["coordinate"]["y"])
    if (
        grid_x >= 0
        and grid_x < grid_size - subgrid_size
        and grid_y >= 0
        and grid_y < grid_size - subgrid_size
    ):
        for p in range(nr_correlations):
            grid[p] = np.asarray(
                _add_one_subgrid(
                    jnp.asarray(grid[p]),
                    jnp.asarray(subgrids[s, p]),
                    jnp.asarray(phasor),
                    grid_x,
                    grid_y,
                    subgrid_size,
                )
            )
