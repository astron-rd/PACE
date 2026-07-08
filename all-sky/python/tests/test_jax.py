"""Testing example / placeholder"""
import copy
from importlib.resources import files
from unittest import TestCase

import numpy as np

from all_sky_python.all_sky_jax import sky_imager_jax_ravel_real_jit
from tests.benchmark import measure_imager
from tests.settings import BENCH_SETTINGS_SINGLE, BENCH_SETTINGS_MANY
from tests.verify import verify_imager


class TestAllSkyImagingJax(TestCase):
    def test_all_sky_ravel_real_image_verify(self):
        """Verify the all sky imager against reference images"""

        baselines = np.load(files("tests.data").joinpath("baselines.npy"))
        visibilities = np.load(files("tests.data").joinpath("visibilities.npy"))

        for i in [16, 32, 64, 95, 205, 256, 512]:
            verify_imager(
                sky_imager_jax_ravel_real_jit,
                BENCH_SETTINGS_SINGLE,
                baselines,
                visibilities[0],
            )

    def test_bench_all_sky_jax_ravel_real_256_256_single(self):
        """Benchmark, repeated measure single visibility, 256x256"""

        settings = copy.copy(BENCH_SETTINGS_SINGLE)
        settings.name = "All Sky Imager Jax Ravel Real; single visibility 256x256"
        measure_imager(sky_imager_jax_ravel_real_jit, settings)

    def test_bench_all_sky_jax_ravel_real_256_256_many(self):
        """Benchmark, 10 visibilities, 256x256"""

        settings = copy.copy(BENCH_SETTINGS_MANY)
        settings.name = "All Sky Imager Jax Ravel Real; many visibilities 256x256"
        measure_imager(sky_imager_jax_ravel_real_jit, settings)
