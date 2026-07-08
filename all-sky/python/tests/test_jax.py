"""JAX All-sky image tests + benchmarks"""

import copy
from importlib.resources import files
from unittest import TestCase

import numpy as np

from all_sky_python.all_sky_jax import sky_imager_jax_ravel_real_jit
from tests.benchmark import measure_imager
from tests.settings import BENCH_SETTINGS_MANY, BENCH_SETTINGS_SINGLE
from tests.verify import verify_imager


class TestAllSkyImagingJax(TestCase):
    def test_all_sky_ravel_real_image_verify(self):
        """Verify the all sky imager against reference images"""

        visibilities = np.load(files("tests.data").joinpath("visibilities.npy"))
        baselines = np.load(files("tests.data").joinpath("baselines.npy"))

        settings = copy.copy(BENCH_SETTINGS_SINGLE)

        for i in [16, 32, 64, 95, 205, 256, 512]:
            settings.image_size_x = i
            settings.image_size_y = i
            verify_imager(
                sky_imager_jax_ravel_real_jit,
                settings,
                visibilities=visibilities[0],
                baselines=baselines,
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
