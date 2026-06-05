from dataclasses import dataclass

from tests.measurements.data import Settings


@dataclass
class AllSkySettings(Settings):
    image_size_x: int = 256
    image_size_y: int = 256
    frequency: float = 58593750.0
    visibilities_path: str = "tests.data/visibilities.npy"
    baselines_path: str = "tests.data/baselines.npy"
