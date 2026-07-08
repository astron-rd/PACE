import copy
from importlib.resources import files
from typing import Callable

import numpy as np

from tests.load import load_npy
from tests.settings import AllSkySettings


def verify_imager(
    fn: Callable,
    settings: AllSkySettings,
    visibilities=None,
    baselines=None,
):
    """Compare against precomputed stored results

    :warning: This method relies on L, M, and N being derived in the same way across
              implementations to have a meaningful comparison.
    """

    # Prevent modifying original reference
    settings = copy.copy(settings)
    x = settings.image_size_x
    y = settings.image_size_y

    if visibilities is None:
        visibilities, _ = load_npy(settings)

    if baselines is None:
        _, baselines = load_npy(settings)

    reference_image = np.load(files("tests.references").joinpath(f"image_{x}_{y}.npy"))

    def result_image(var_x, var_y):
        return fn(visibilities, baselines, [settings.frequency], var_x, var_y)

    result_image = result_image(x, y)

    # Create a circle as mask just below unit length. and remove those results from
    # the evaluation. The (all-sky imaging) computation does not solve beyond the
    # horizon. This boundary can shift slightly due to numerical error between
    # implementations.
    npix_l, npix_m = np.meshgrid(np.linspace(-1, 1, x), np.linspace(1, -1, y))
    c = npix_l**2 + npix_m**2 < 0.99
    reference_image = np.where(c, reference_image, float("nan"))
    result_image = np.where(c, result_image, float("nan"))

    np.testing.assert_allclose(reference_image, result_image, rtol=1e-04)
