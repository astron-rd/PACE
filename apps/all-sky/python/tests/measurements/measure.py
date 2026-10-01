import importlib.util
import logging
import statistics
import time
from typing import Any, Callable

from tests.measurements.data import MeasureData, ResultData, Settings

PMT_AVAILABLE = importlib.util.find_spec("pmt")
if PMT_AVAILABLE:
    import pmt  # type: ignore[import-not-found] # noqa: F401  # pylint: disable=import-error

logger = logging.getLogger()


class Measure:
    def __init__(self, settings: Settings = None, pmt_backends: str = ""):
        self.measures: list[MeasureData] = []

        self.pmt = False
        if PMT_AVAILABLE:
            pmt_backends = [x.strip() for x in pmt_backends.split(",")]
            logger.info("Using PMT power monitoring")
            self.pms = []
            for pmt_backend in pmt_backends:
                logger.info("Configuring PMT backend: %s", pmt_backend)
                self.pms.append(pmt.create(pmt_backend))
            self.pmt = True
        elif len(pmt_backends) != 0:
            logger.warning("PMT not available!, can't configure backends")

        if settings is None:
            settings = Settings()
        self.settings = settings

    def _add_measure(self, seconds: float, joules: float = 0.0, watts: float = 0.0):
        self.measures.append(MeasureData(seconds=seconds, joules=joules, watts=watts))

    def _accumalate_pmt_reads(self) -> list[Any]:
        """PMT helper function to support multiple backends in single measurement"""
        return [x.read() for x in self.pms]

    def _accumulate_pmt_totals(self, starts: list, ends: list):
        """PMT helper function to accumulate totals of multiple backends"""
        return (
            [pmt.joules(a, b) for a, b in zip(starts, ends, strict=True)],
            [pmt.watts(a, b) for a, b in zip(starts, ends, strict=True)],
        )

    def _run_time(self, fn: Callable[[], None]):
        """Measure performance of fn, only time"""
        for _ in range(self.settings.iterations):
            start_time = time.time()
            result = fn()
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            self._add_measure(time.time() - start_time)

    def _run_pmt_rapl(self, fn: Callable[[], None]):
        """Measure performance of fn with PMT integration for power measurements"""
        for _ in range(self.settings.iterations):
            starts = self._accumalate_pmt_reads()
            result = fn()
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            ends = self._accumalate_pmt_reads()
            power_data = self._accumulate_pmt_totals(starts, ends)
            self._add_measure(
                pmt.seconds(starts[0], ends[0]), sum(power_data[0]), sum(power_data[1])
            )

    def run(self, fn: Callable[[], None]):
        """Run the function under test and measure it

        Measurements are stored in self.measures
        """
        if self.pmt:
            self._run_pmt_rapl(fn)
        else:
            self._run_time(fn)

    def warmup(self, fn: Callable[[], None]):
        """Do warmup runs"""

        for _ in range(self.settings.warmup):
            fn()

    def compute(self, measure_attribute: str = "seconds") -> ResultData:
        """Compute mean, std, min, max over a given attribute (seconds, joules watts)"""
        result = ResultData()
        result.mean = statistics.mean(
            [getattr(x, measure_attribute) for x in self.measures]
        )
        result.std = statistics.stdev(
            [getattr(x, measure_attribute) for x in self.measures]
        )
        for measure in self.measures:
            if getattr(measure, measure_attribute) < result.min:
                result.min = getattr(measure, measure_attribute)
            if getattr(measure, measure_attribute) > result.max:
                result.max = getattr(measure, measure_attribute)
        return result
