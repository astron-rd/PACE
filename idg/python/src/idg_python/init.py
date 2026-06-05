import numpy as np

from .kernels.numba import evaluate_spheroidal


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
    x = np.abs(np.linspace(-1, 1, num=subgrid_size, endpoint=False))
    x_spheroidal = evaluate_spheroidal(x)

    # Construct 2D taper array
    taper = x_spheroidal[np.newaxis, :] * x_spheroidal[:, np.newaxis]

    # Cast to correct type
    return taper.astype(np.float32)
