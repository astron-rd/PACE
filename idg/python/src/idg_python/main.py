import json
import time

import h5py
import numpy as np

from .config import settings
from .idg import FOURIER_DOMAIN_TO_IMAGE_DOMAIN, Gridder
from .init import get_taper

GRIDTYPE = np.complex64


class Timer:
    """Times operations and collects their durations keyed by description."""

    def __init__(self):
        self.timings = {}

    @staticmethod
    def print_header(title, header_length=50, newline="\n"):
        print(newline + "=" * header_length)
        print(title)
        print("=" * header_length)

    def time(self, description, operation):
        print(f"{description:<38}", end="")
        start = time.time()
        result = operation()
        end = time.time()
        duration = end - start
        print(f" {duration:>9.6f} s")
        self.timings[description] = duration
        return result


def main():
    timer = Timer()

    timer.print_header("READING INPUT DATA", newline="")
    with h5py.File(settings.input, "r") as infile:
        grid_size = int(infile.attrs["grid_size"])
        subgrid_size = int(infile.attrs["subgrid_size"])

        uvw_ds = infile["uvws"]
        if not isinstance(uvw_ds, h5py.Dataset):
            print("Invalid input file: uvws is not defined")
            return
        uvw = timer.time("Load UVW coordinates", lambda: uvw_ds[...])

        frequencies_ds = infile["frequencies"]
        if not isinstance(frequencies_ds, h5py.Dataset):
            print("Invalid input file: frequencies is not defined")
            return
        frequencies = timer.time("Load frequencies", lambda: frequencies_ds[...])

        metadata_ds = infile["metadata"]
        if not isinstance(metadata_ds, h5py.Dataset):
            print("Invalid input file: metadata is not defined")
            return
        metadata = timer.time("Load metadata", lambda: metadata_ds[...])
        nr_subgrids = metadata.shape[0]

        visibilities_ds = infile["visibilities"]
        if not isinstance(visibilities_ds, h5py.Dataset):
            print("Invalid input file: visibilities is not defined")
            return
        visibilities = timer.time("Load visibilities", lambda: visibilities_ds[...])

    wavenumbers = (frequencies * 2 * np.pi) / settings.speed_of_light
    image_size = settings.speed_of_light / frequencies[-1]
    nr_correlations_in = visibilities.shape[-1]

    parameters = {
        "nr_correlations_in": nr_correlations_in,
        "nr_correlations_out": settings.nr_correlations_out,
        "nr_channels": len(frequencies),
        "nr_timesteps": uvw.shape[1],
        "nr_baselines": uvw.shape[0],
        "subgrid_size": subgrid_size,
        "grid_size": grid_size,
    }

    timer.print_header("PARAMETERS")
    for key, value in parameters.items():
        print(f"{key:<39} {value:>10}")

    grid = np.zeros(
        (settings.nr_correlations_out, grid_size, grid_size), dtype=GRIDTYPE
    )

    taper = timer.time("Initialize taper", lambda: get_taper(subgrid_size=subgrid_size))

    subgrids = np.zeros(
        shape=(nr_subgrids, settings.nr_correlations_out, subgrid_size, subgrid_size),
        dtype=GRIDTYPE,
    )

    gridder = timer.time(
        "Initialize gridder",
        lambda: Gridder(
            nr_correlations_in=nr_correlations_in,
            subgrid_size=subgrid_size,
        ),
    )

    timer.print_header("MAIN")

    timer.time(
        "Grid visibilities",
        lambda: gridder.grid_onto_subgrids(
            w_step=settings.w_step,
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

    timer.time(
        "Add subgrids",
        lambda: gridder.add_subgrids_to_grid(
            metadata=metadata, subgrids=subgrids, grid=grid
        ),
    )

    timer.time(
        "Transform grid",
        lambda: gridder.transform(direction=FOURIER_DOMAIN_TO_IMAGE_DOMAIN, grid=grid),
    )

    timer.print_header("TIMINGS")
    total_time = sum(timer.timings.values())
    for operation, duration in timer.timings.items():
        percentage = (duration / total_time) * 100
        print(f"{operation:<30} {duration:>8.3f} s ({percentage:>5.1f}%)")
    print(f"{'Total':<30} {total_time:>8.3f} s")

    if settings.store:
        with h5py.File("output.h5", "w") as output_file:
            output_file.create_dataset("grid", data=grid)
            output_file.create_dataset("subgrids", data=subgrids)
    if settings.json_output:
        output = {"parameters": {}, "timings": {}}

        for key, value in parameters.items():
            output["parameters"][key] = value

        for operation, duration in timer.timings.items():
            output["timings"][operation.lower().replace(" ", "_")] = round(
                duration * 1000, 2
            )

        with open(settings.json_output, "w") as f:
            json.dump(output, f, indent=2)
