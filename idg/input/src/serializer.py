import numpy as np
from pathlib import Path

DIRECTORY: Path = Path("artifacts")


class Serializer:
    def __init__(self):
        DIRECTORY.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, array: np.ndarray) -> None:
        np.save(DIRECTORY.joinpath(name + ".npy"), array)
