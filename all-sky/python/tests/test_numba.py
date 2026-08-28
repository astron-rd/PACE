"""Numba All-sky image tests + benchmarks"""

import copy
from importlib.resources import files

import numpy as np
import pytest

from all_sky_python.all_sky_numba import sky_imager_numba_ravel_real
from tests.base import BaseTestCase
from tests.benchmark import measure_imager
from tests.settings import BENCH_SETTINGS_MANY, BENCH_SETTINGS_SINGLE
from tests.verify import verify_imager


@pytest.fixture(scope="class")
def pmt(request):
    """Make PMT configuration available to test cases under self.pmt"""
    request.cls.pmt = request.config.option.pmt


@pytest.mark.usefixtures("pmt")
class TestAllSkyImagingNumba(BaseTestCase):
    def test_all_sky_ravel_real_image_verify(self):
        """Verify the all sky imager against reference images"""

        visibilities = np.load(files("tests.data").joinpath("visibilities.npy"))
        baselines = np.load(files("tests.data").joinpath("baselines.npy"))

        settings = copy.copy(BENCH_SETTINGS_SINGLE)

        for i in [16, 32, 64, 95, 205, 256]:
            settings.image_size_x = i
            settings.image_size_y = i
            verify_imager(
                sky_imager_numba_ravel_real,
                settings,
                visibilities[0],
                baselines,
            )

    def test_bench_all_sky_numba_ravel_real_256_256_single(self):
        """Benchmark, repeated measure single visibility, 256x256"""

        settings = copy.copy(BENCH_SETTINGS_SINGLE)
        settings.name = "All Sky Imager Numba Ravel Real; single visibility 256x256"
        measure_imager(sky_imager_numba_ravel_real, settings, self.pmt)

    def test_bench_all_sky_numba_ravel_real_256_256_many(self):
        """Benchmark, 10 visibilities, 256x256"""

        settings = copy.copy(BENCH_SETTINGS_MANY)
        settings.name = "All Sky Imager Numba Ravel Real; many visibilities 256x256"
        measure_imager(sky_imager_numba_ravel_real, settings, self.pmt)
