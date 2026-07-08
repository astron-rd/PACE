import logging
from typing import Callable

from tests.load import load_npy
from tests.measurements.measure import Measure
from tests.settings import AllSkySettings

logger = logging.getLogger()


def measure_imager(fn: Callable, settings: AllSkySettings):
    visibilities, baselines = load_npy(settings)

    measure = Measure(settings)
    measure.warmup(
        lambda var_x, var_y: fn(
            visibilities[0],
            baselines,
            settings.frequency,
            var_x,
            var_y,
        )
    )

    for x in range(settings.variances):
        measure.run(
            lambda var_x, var_y: fn(
                visibilities[x % len(visibilities)],  # noqa (false positive)
                baselines,
                settings.frequency,
                var_x,
                var_y,
            )
        )
    results = {
        "Time": measure.compute("seconds"),
        "Joules": measure.compute("joules"),
        "Watts": measure.compute("watts"),
    }
    for measurement, result in results.items():
        logger.info(
            "[%s][%s]: min: %.4f max: %.4f, mean: %.4f, stddev: %.4f",
            measurement,
            settings.name,
            result.min,
            result.max,
            result.mean,
            result.std,
        )
