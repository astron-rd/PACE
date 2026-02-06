#  Copyright (C) 2026 ASTRON (Netherlands Institute for Radio Astronomy)
#  Copyright (C) 2025 Corne Lukken
#  SPDX-License-Identifer: Apache-2.0

import numpy as np
import numba
from nptyping import NDArray, Shape, Float64, Complex64
from numba import prange

SPEED_OF_LIGHT = 299792458.0


@numba.njit(parallel=True, fastmath=True)
def mat_vec_broadcast_mul(M, v):
    p, q, _ = M.shape  # 96, 96, 1
    n = v.shape[0]  # 51040
    out = np.empty((p, q, n), dtype=M.dtype)

    for i in prange(p):
        for j in range(q):
            scalar = M[i, j, 0]
            for k in range(n):
                out[i, j, k] = scalar * v[k]
    return out


@numba.njit(parallel=True, fastmath=True)
def mat_mat_broadcast_sum(M, N):
    p, q, r = M.shape  # 96, 96, 51040
    n = N.shape  # 92, 96, 1
    out = np.empty((p, q, r), dtype=M.dtype)

    for i in prange(p):
        for j in range(q):
            scalar = N[i, j, 0]
            for k in range(r):
                out[i, j, k] = scalar + M[i, j, k]
    return out


@numba.njit(parallel=True, fastmath=True)
def mat_mat_broadcast_mul(M, N):
    p, q, r = M.shape  # 96, 96, 51040
    n = N.shape  # 92, 96, 1
    out = np.empty((p, q, r), dtype=M.dtype)

    for i in prange(p):
        for j in range(q):
            scalar = N[i, j, 0]
            for k in range(r):
                out[i, j, k] = scalar * M[i, j, k]
    return out


@numba.njit(parallel=True, fastmath=True)
def mat_scalar_loop(M, s):
    p, q, r = M.shape  # 96, 96, 51040
    out = np.empty((p, q, r), dtype=M.dtype)
    for i in prange(p):
        for j in range(q):
            for k in range(r):
                out[i, j, k] = (s * M[i, j, k])[0]
    return out


# Do not jit, worse performance
def sky_imager_numba_ravel_real(
    visibilities: NDArray[Shape["Dim, Dim"], Complex64],
    baselines: NDArray[Shape["Dim, Dim, 3"], Float64],
    freq: NDArray[Shape["1"], Float64],
    npix_l: int,
    npix_m: int,
):
    """
    :param visibilities: 2d rectangular array of visibilities
    :param baselines: 3d array with u, v, w per antenna baseline (N^2)
    :param freq: the frequency in hertz
    :param npix_l: number of pixels length
    :param npix_m: number of pixels height
    :return: 2d image from the imaging process
    """

    grid_l = np.zeros((npix_l, npix_m), dtype=np.float32)
    grid_m = np.zeros((npix_m, npix_l), dtype=np.float32)
    npix_l = np.linspace(-1, 1, npix_l)
    npix_m = np.linspace(1, -1, npix_m)
    for x in prange(npix_l):
        for y in range(npix_m):
            grid_l[x][y] = npix_l[y]

    for x in prange(npix_m):
        for y in range(npix_l):
            grid_m[y][x] = npix_m[y]

    # Select and ravel
    c = grid_l**2 + grid_m**2 < 1  # Create unit circle 2D image
    lt = np.zeros(np.sum(c))
    mt = np.zeros(np.sum(c))
    idx = 0
    for x, m in enumerate(c):
        for y, l in enumerate(m):
            if l:
                lt[idx] = grid_l[x][y]
                idx += 1
    idx = 0
    for x, m in enumerate(c):
        for y, l in enumerate(m):
            if l:
                mt[idx] = grid_m[x][y]
                idx += 1
    nt = np.sqrt(1 - lt**2 - mt**2)

    u, v, w = baselines.astype("float32").T
    prod = (
        mat_vec_broadcast_mul(u[:, :, np.newaxis], lt)
        + mat_vec_broadcast_mul(v[:, :, np.newaxis], mt)
        + mat_vec_broadcast_mul(w[:, :, np.newaxis], (nt - 1))
    )
    phase = -2 * np.pi * mat_scalar_loop(prod, freq) / SPEED_OF_LIGHT
    vf = np.angle(visibilities[:, :, np.newaxis])
    vr = np.abs(visibilities[:, :, np.newaxis])
    pr = mat_mat_broadcast_mul(np.cos(mat_mat_broadcast_sum(phase, vf)), vr)

    img_dim = grid_l.shape[0] * grid_l.shape[1]
    img = np.full(img_dim, 0, dtype="float32")
    retainer = np.zeros(pr.shape[2])
    for z in prange(pr.shape[2]):
        retainer[z] = np.mean(pr[:, :, z])
    img[c.ravel()] = retainer

    return img.reshape(grid_l.shape)