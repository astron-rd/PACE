import os
import numpy as np

DIRECTORY = "artifacts"


class Serializer:
    def __init__(self):
        os.makedirs(DIRECTORY, exist_ok=True)

    def save(self, name: str, array: np.ndarray) -> None:
        np.save(DIRECTORY + "/" + name + ".npy", array)
