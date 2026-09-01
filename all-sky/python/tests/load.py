from importlib.resources import files

import numpy as np

from tests.settings import AllSkySettings


def load_npy(settings: AllSkySettings) -> (np.ndarray, np.ndarray):
    path, file = settings.visibilities_path.rsplit("/", 1)
    visibilities = np.load(files(path).joinpath(file))

    path, file = settings.baselines_path.rsplit("/", 1)
    baselines = np.load(files(path).joinpath(file))

    return (visibilities, baselines)
