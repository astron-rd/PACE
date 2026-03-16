import jax
from nptyping import NDArray, Shape, Float64, Float32, Bool, Complex64
from jax import numpy as jnp

from all_sky_python.constants import SPEED_OF_LIGHT

# https://github.com/astron-rd/PACE/wiki/Jax
# https://docs.jax.dev/en/latest/installation.html
# https://docs.jax.dev/en/latest/config_options.html#jax_platforms
jax.config.update("jax_platform_name", "cpu")
# Gives another ~60% speedup on CPU
jax.config.update("jax_exec_time_optimization_effort", 1.0)


def _sky_imager_jax_ravel_real_jit(
    visibilities: NDArray[Shape["Dim, Dim"], Complex64],
    baselines: NDArray[Shape["Dim, Dim, 3"], Float64],
    freq: NDArray[Shape["1"], Float64],
    lt: NDArray[Shape["Ravel"], Float32],
    mt: NDArray[Shape["Ravel"], Float32],
):
    """
    Produce a linear array representing a unit circle on a 2D image from correlated
    visibilities and antenna baselines, only computes the real component.

    Takes two linear arrays lt and mt as argument that are the linear mapping of the
    unit circle drawn unto the original square image.
    """
    visibilities = jnp.array(visibilities)
    freq = jnp.array(freq)
    nt = jnp.sqrt(1 - lt**2 - mt**2)

    u, v, w = jnp.array(baselines.astype("float32")).T
    prod = (
        u[:, :, jnp.newaxis] * lt
        + v[:, :, jnp.newaxis] * mt
        + w[:, :, jnp.newaxis] * (nt - 1)
    )
    phase = -2 * jnp.pi * freq * prod / SPEED_OF_LIGHT
    vis = visibilities[:, :, jnp.newaxis]

    return jnp.mean(jnp.cos(phase + jnp.angle(vis)) * jnp.abs(vis), axis=(0, 1))


f_imager_ravel_real = jax.jit(_sky_imager_jax_ravel_real_jit)  # , static_argnums=(3,))


def sky_imager_jax_ravel_real_jit(
    visibilities: NDArray[Shape["Dim, Dim"], Complex64],
    baselines: NDArray[Shape["Dim, Dim, 3"], Float64],
    freq: NDArray[Shape["1"], Float64],
    npix_l: int,
    npix_m: int,
):
    npix_l, npix_m = jnp.meshgrid(
        jnp.linspace(-1, 1, npix_l), jnp.linspace(1, -1, npix_m)
    )
    c = npix_l**2 + npix_m**2 < 1
    lt, mt = npix_l[c].ravel(), npix_m[c].ravel()
    img = jnp.full(jnp.prod(jnp.array(npix_l.shape)), jnp.nan, dtype="float32")

    # Note, Jax arrays are immutable, for every xxx.at[].set() a full
    # allocation / deallocation occurs. These are extremely costly and must be minimized
    img = img.at[c.ravel()].set(
        f_imager_ravel_real(visibilities, baselines, freq, lt, mt)
    )

    return img.reshape(npix_l.shape)
