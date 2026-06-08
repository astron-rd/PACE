import logging
from importlib.resources import files
from typing import Callable

import numpy as np

from tests.measurements.data import Settings
from tests.measurements.measure import Measure

logger = logging.getLogger()


def _load_npy(settings: Settings) -> (np.ndarray, np.ndarray):
    path, file = settings.visibilities_path.rsplit("/", 1)
    visibilities = np.load(files(path).joinpath(file))

    path, file = settings.baselines_path.rsplit("/", 1)
    baselines = np.load(files(path).joinpath(file))

    return (visibilities, baselines)


def measure_imager(fn: Callable, settings: Settings):
    visibilities, baselines = _load_npy(settings)

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
