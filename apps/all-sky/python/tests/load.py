from importlib.resources import files

import h5py
import numpy as np

from tests.settings import AllSkySettings


def load_hdf5(settings: AllSkySettings) -> (np.ndarray, np.ndarray):
    path, file = settings.test_data_path.rsplit("/", 1)
    test_data = h5py.File(files(path).joinpath(file), "r")
    visibilities = test_data["visibilities"][...]
    baselines = test_data["baselines"][...]

    return (visibilities, baselines)
