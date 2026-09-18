import h5py
import numpy as np

data = np.load("tests/references/image_512_512.npy")

print(data)

with h5py.File("tests/references/image_512_512.h5", "w") as outfile:
    outfile.create_dataset("data", data=data)
