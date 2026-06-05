import argparse
import json
import time

import h5py
import numpy as np

from idg_python import idgtypes
from idg_python.idg import Gridder
from idg_python.init import get_taper

# Dictionary to store all timings
timings = {}

# Constants
NR_CORRELATIONS_OUT = 1  # I
W_STEP = 1.0  # w step in wavelengths
SPEED_OF_LIGHT = 299792458.0


def main():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input HDF5 file")
    parser.add_argument(
        "--store", action="store_true", default=False, help="Store data in HDF5 format"
    )
    parser.add_argument(
        "--json",
        const="timings.json",
        nargs="?",
        help="Output timings in JSON format (optional: specify filename)",
    )
    args = parser.parse_args()

    def print_header(title, header_length=50, newline="\n"):
        print(newline + "=" * header_length)
        print(title)
        print("=" * header_length)

    def timeit(description, operation):
        print(f"{description:<38}", end="")
        start = time.time()
        result = operation()
        end = time.time()
        duration = end - start
        print(f" {duration:>9.6f} s")
        timings[description] = duration
        return result

    print_header("READING INPUT DATA", newline="")
    with h5py.File(args.input, "r") as infile:
        grid_size = int(infile.attrs["grid_size"])
        subgrid_size = int(infile.attrs["subgrid_size"])

        uvw_ds = infile["uvws"]
        if not isinstance(uvw_ds, h5py.Dataset):
            print("Invalid input file: uvws is not defined")
            return
        uvw = timeit("Load UVW coordinates", lambda: uvw_ds[...])

        frequencies_ds = infile["frequencies"]
        if not isinstance(frequencies_ds, h5py.Dataset):
            print("Invalid input file: frequencies is not defined")
            return
        frequencies = timeit("Load frequencies", lambda: frequencies_ds[...])

        metadata_ds = infile["metadata"]
        if not isinstance(metadata_ds, h5py.Dataset):
            print("Invalid input file: metadata is not defined")
            return
        metadata = timeit("Load metadata", lambda: metadata_ds[...])
        nr_subgrids = metadata.shape[0]

        visibilities_ds = infile["visibilities"]
        if not isinstance(visibilities_ds, h5py.Dataset):
            print("Invalid input file: visibilities is not defined")
            return
        visibilities = timeit("Load visibilities", lambda: visibilities_ds[...])

    wavenumbers = (frequencies * 2 * np.pi) / SPEED_OF_LIGHT
    image_size = SPEED_OF_LIGHT / frequencies[-1]
    nr_correlations_in = visibilities.shape[-1]

    parameters = {
        "nr_correlations_in": nr_correlations_in,
        "nr_correlations_out": NR_CORRELATIONS_OUT,
        "nr_channels": len(frequencies),
        "nr_timesteps": uvw.shape[1],
        "nr_baselines": uvw.shape[0],
        "subgrid_size": subgrid_size,
        "grid_size": grid_size,
    }

    print_header("PARAMETERS")
    for key, value in parameters.items():
        print(f"{key:<39} {value:>10}")

    grid = np.zeros(
        (NR_CORRELATIONS_OUT, grid_size, grid_size), dtype=idgtypes.gridtype
    )

    taper = timeit("Initialize taper", lambda: get_taper(subgrid_size=subgrid_size))

    subgrids = np.zeros(
        shape=(nr_subgrids, NR_CORRELATIONS_OUT, subgrid_size, subgrid_size),
        dtype=idgtypes.gridtype,
    )

    gridder = timeit(
        "Initialize gridder",
        lambda: Gridder(
            nr_correlations_in=nr_correlations_in,
            subgrid_size=subgrid_size,
        ),
    )

    print_header("MAIN")

    timeit(
        "Grid visibilities",
        lambda: gridder.grid_onto_subgrids(
            w_step=W_STEP,
            image_size=image_size,
            grid_size=grid_size,
            wavenumbers=wavenumbers,
            uvw=uvw,
            visibilities=visibilities,
            taper=taper,
            metadata=metadata,
            subgrids=subgrids,
        ),
    )

    timeit(
        "Add subgrids",
        lambda: gridder.add_subgrids_to_grid(
            metadata=metadata, subgrids=subgrids, grid=grid
        ),
    )

    timeit(
        "Transform grid",
        lambda: gridder.transform(
            direction=idgtypes.FOURIER_DOMAIN_TO_IMAGE_DOMAIN, grid=grid
        ),
    )

    print_header("TIMINGS")
    total_time = sum(timings.values())
    for operation, duration in timings.items():
        percentage = (duration / total_time) * 100
        print(f"{operation:<30} {duration:>8.3f} s ({percentage:>5.1f}%)")
    print(f"{'Total':<30} {total_time:>8.3f} s")

    if args.store:
        with h5py.File("output.h5", "w") as output_file:
            output_file.create_dataset("grid", data=grid)
            output_file.create_dataset("subgrids", data=subgrids)
    if args.json:
        output = {"parameters": {}, "timings": {}}

        for key, value in parameters.items():
            output["parameters"][key] = value

        for operation, duration in timings.items():
            output["timings"][operation.lower().replace(" ", "_")] = round(
                duration * 1000, 2
            )

        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
