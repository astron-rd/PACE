import numpy as np
from nptyping import NDArray, Shape, Float64, Complex64
import numba
from numba import set_num_threads
from numba import prange

from all_sky_python.constants import SPEED_OF_LIGHT

numba.config.THREADING_LAYER_PRIORITY = ["omp", "tbb", "workqueue"]
set_num_threads(10)  # Set two the number of physical cores to not use SMT,
# TODO: Automate set_num_threads amount


@numba.njit(parallel=True, fastmath=True, nogil=True)
def mat_vec_broadcast_mul(m, v):
    """
    m shape: num_ant, num_ant, 1
    v length: pixels_x*y
    """
    p, q, _ = m.shape
    n = v.shape[0]
    out = np.empty((p, q, n), dtype=m.dtype)

    for i in prange(p):
        for j in range(q):
            scalar = m[i, j, 0]
            for k in range(n):
                out[i, j, k] = scalar * v[k]
    return out


@numba.njit(parallel=True, fastmath=True, nogil=True)
def mat_mat_broadcast_sum(m, n):
    """
    m shape: num_ant, num_ant, pixels_x*y
    n shape: num_ant, num_ant, 1
    """
    p, q, r = m.shape
    out = np.empty((p, q, r), dtype=m.dtype)

    for i in prange(p):
        for j in range(q):
            scalar = n[i, j, 0]
            for k in range(r):
                out[i, j, k] = scalar + m[i, j, k]
    return out


@numba.njit(parallel=True, fastmath=True, nogil=True)
def mat_mat_broadcast_mul(m, n):
    """
    m shape: num_ant, num_ant, pixels_x*y
    n shape: num_ant, num_ant, 1
    """
    p, q, r = m.shape
    out = np.empty((p, q, r), dtype=m.dtype)

    for i in prange(p):
        for j in range(q):
            scalar = n[i, j, 0]
            for k in range(r):
                out[i, j, k] = scalar * m[i, j, k]
    return out


@numba.njit(parallel=True, fastmath=True, nogil=True)
def mat_scalar_loop(m, s):
    """
    m shape: num_ant, num_ant, pixels_x*y
    """
    p, q, r = m.shape
    out = np.empty((p, q, r), dtype=m.dtype)
    for i in prange(p):
        for j in range(q):
            for k in range(r):
                out[i, j, k] = (s * m[i, j, k])[0]
    return out


@numba.njit(parallel=True, fastmath=True, nogil=True)
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

    # Gridding without meshgrid, not support by numba when jitting
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
    mt = np.zeros(lt.size)
    idx = 0
    for x in range(c.shape[0]):
        for y in range(c.shape[1]):
            if c[x][y]:
                lt[idx] = grid_l[x][y]
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
    vis_angle = np.angle(visibilities[:, :, np.newaxis])
    vis_abs = np.abs(visibilities[:, :, np.newaxis])
    euler_phase_vis = mat_mat_broadcast_mul(
        np.cos(mat_mat_broadcast_sum(phase, vis_angle)), vis_abs
    )

    img_dim = grid_l.shape[0] * grid_l.shape[1]
    img = np.full(img_dim, 0, dtype="float32")

    image_retainer = np.zeros(euler_phase_vis.shape[2])
    for z in prange(euler_phase_vis.shape[2]):
        image_retainer[z] = np.mean(euler_phase_vis[:, :, z])
    img[c.ravel()] = image_retainer

    return img.reshape(grid_l.shape)