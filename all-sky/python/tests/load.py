from importlib.resources import files

import h5py
import numpy as np

from tests.settings import AllSkySettings


def load_hdf5(settings: AllSkySettings) -> (np.ndarray, np.ndarray):
    path, file = settings.visibilities_path.rsplit("/", 1)
    visibilities = h5py.File(files(path).joinpath(file), "r")["data"][...]

    path, file = settings.baselines_path.rsplit("/", 1)
    baselines = h5py.File(files(path).joinpath(file), "r")["data"][...]

    return (visibilities, baselines)
