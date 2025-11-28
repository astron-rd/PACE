import argparse
import time
import numpy as np

import idgtypes
from init import get_uvw, get_metadata, get_frequencies, get_visibilities, get_taper
from idg import Gridder  # type: ignore

# Dictionary to store all timings
timings = dict()

# Constants
NR_CORRELATIONS_IN = 2  # XX, YY
NR_CORRELATIONS_OUT = 1  # I
W_STEP = 1.0  # w step in wavelengths
SPEED_OF_LIGHT = 299792458.0
START_FREQUENCY = 150e6  # 150 MHz
FREQUENCY_INCREMENT = 1e6  # 1 MHz

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--subgrid_size", type=int, default=32, help="Size of the subgrid in pixels"
)
parser.add_argument(
    "--grid_size", type=int, default=1024, help="Size of the grid in pixels"
)
parser.add_argument(
    "--observation_hours",
    type=float,
    default=4,
    help="Length of the observation in hours",
)
parser.add_argument(
    "--nr_channels", type=int, default=16, help="Number of frequency channels"
)
parser.add_argument("--nr_stations", type=int, default=20, help="Number of stations")
parser.add_argument(
    "--store", action="store_true", default=False, help="Store data in Numpy format"
)
args = parser.parse_args()

SUBGRID_SIZE = args.subgrid_size
GRID_SIZE = args.grid_size
OBSERVATION_HOURS = args.observation_hours
NR_CHANNELS = args.nr_channels
STORE_DATA = args.store
OUTPUT_JSON = args.json

# Derived arguments
NR_TIMESTEPS = int(OBSERVATION_HOURS * 3600)
END_FREQUENCY = START_FREQUENCY + NR_CHANNELS * FREQUENCY_INCREMENT
IMAGE_SIZE = SPEED_OF_LIGHT / END_FREQUENCY
NR_STATIONS = args.nr_stations
NR_BASELINES = NR_STATIONS * (NR_STATIONS - 1) // 2


def print_header(title, header_length=50, newline="\n"):
    print(newline + "=" * header_length)
    print(title)
    print("=" * header_length)


params = {
    "nr_correlations_in": NR_CORRELATIONS_IN,
    "nr_correlations_out": NR_CORRELATIONS_OUT,
    "start_frequency": f"{START_FREQUENCY*1e-6} MHz",
    "frequency_increment": f"{FREQUENCY_INCREMENT*1e-6} MHz",
    "nr_channels": NR_CHANNELS,
    "nr_timesteps": NR_TIMESTEPS,
    "nr_stations": NR_STATIONS,
    "nr_baselines": NR_BASELINES,
    "subgrid_size": SUBGRID_SIZE,
    "grid_size": GRID_SIZE,
}

print_header("PARAMETERS", newline="")
for key, value in params.items():
    print(f"{key:<39} {value:>10}")


def timeit(description, operation):
    print(f"{description:<40}", end="")
    start = time.time()
    result = operation()
    end = time.time()
    duration = end - start
    if duration > 1:
        print(f" {duration:>7.3f} s")
    else:
        print(f" {duration*1e3:>6.3f} ms")
    timings[description] = duration
    return result


print_header("INITIALIZATION")
uvw = timeit(
    "Initialize UVW coordinates",
    lambda: get_uvw(
        observation_hours=OBSERVATION_HOURS,
        nr_baselines=NR_BASELINES,
        grid_size=GRID_SIZE,
    ),
)

frequencies = timeit(
    "Initialize frequencies",
    lambda: get_frequencies(START_FREQUENCY, FREQUENCY_INCREMENT, NR_CHANNELS),
)
wavenumbers = (frequencies * 2 * np.pi) / SPEED_OF_LIGHT

metadata = timeit(
    "Initialize metadata",
    lambda: get_metadata(
        nr_channels=NR_CHANNELS,
        subgrid_size=SUBGRID_SIZE,
        grid_size=GRID_SIZE,
        uvw=uvw,
    ),
)
nr_subgrids = metadata.shape[0]

visibilities = timeit(
    "Initialize visibilities",
    lambda: get_visibilities(
        nr_correlations=NR_CORRELATIONS_IN,
        nr_channels=NR_CHANNELS,
        nr_timesteps=NR_TIMESTEPS,
        nr_baselines=NR_BASELINES,
        image_size=IMAGE_SIZE,
        grid_size=GRID_SIZE,
        frequencies=frequencies,
        uvw=uvw,
    ),
)

grid = np.zeros((NR_CORRELATIONS_OUT, GRID_SIZE, GRID_SIZE), dtype=idgtypes.gridtype)

taper = timeit("Initialize taper", lambda: get_taper(subgrid_size=SUBGRID_SIZE))

subgrids = np.zeros(
    shape=(nr_subgrids, NR_CORRELATIONS_OUT, SUBGRID_SIZE, SUBGRID_SIZE),
    dtype=idgtypes.gridtype,
)

gridder = timeit(
    "Initialize gridder",
    lambda: Gridder(
        nr_correlations_in=NR_CORRELATIONS_IN,
        subgrid_size=SUBGRID_SIZE,
    ),
)

print_header("MAIN")

timeit(
    "Grid visibilities",
    lambda: gridder.grid_onto_subgrids(
        w_step=W_STEP,
        image_size=IMAGE_SIZE,
        grid_size=GRID_SIZE,
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
        metadata=metadata,
        subgrids=subgrids,
        grid=grid,
    ),
)

timeit(
    "Transform grid",
    lambda: gridder.transform(
        direction=idgtypes.FOURIER_DOMAIN_TO_IMAGE_DOMAIN,
        grid=grid,
    ),
)

print_header("TIMINGS")
total_time = sum(timings.values())
for operation, duration in timings.items():
    percentage = (duration / total_time) * 100
    print(f"{operation:<30} {duration:>8.2f} s ({percentage:>5.1f}%)")
print(f"{'Total':<30} {total_time:>8.2f} s")

if STORE_DATA:
    np.save("uvw.npy", uvw)
    np.save("frequencies.npy", frequencies)
    np.save("taper.npy", taper)
    np.save("metadata.npy", metadata)
    np.save("visibilities.npy", visibilities)
    np.save("subgrids.npy", subgrids)
    np.save("grid.npy", grid)
    np.save("image.npy", grid)
