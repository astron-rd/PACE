import jax.numpy as jnp
import numpy as np

from .kernels.spheroidal import evaluate_spheroidal


def get_taper(subgrid_size: int) -> np.ndarray:
    """Construct the subgrid taper by evaluating the prolate spheroidal wave
    function over a 1D grid and taking the outer product (JAX port)."""
    x = np.abs(np.linspace(-1, 1, num=subgrid_size, endpoint=True))
    x_spheroidal = np.asarray(evaluate_spheroidal(jnp.asarray(x)))

    taper = x_spheroidal[np.newaxis, :] * x_spheroidal[:, np.newaxis]
    return taper.astype(np.float32)
