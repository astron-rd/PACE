import argparse
import time
import numpy as np

import idgtypes
from init import get_uvw, get_metadata, get_frequencies, get_visibilities, get_taper
from idg import Gridder  # type: ignore


NR_CORRELATIONS_IN = 2  # XX, YY
NR_CORRELATIONS_OUT = 1  # I
W_STEP = 1.0  # w step in wavelengths
SPEED_OF_LIGHT = 299792458.0
START_FREQUENCY = 150e6  # 150 MHz
FREQUENCY_INCREMENT = 1e6  # 1 MHz

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
NR_TIMESTEPS = int(OBSERVATION_HOURS * 3600)
NR_CHANNELS = args.nr_channels
STORE_DATA = args.store

END_FREQUENCY = START_FREQUENCY + NR_CHANNELS * FREQUENCY_INCREMENT
IMAGE_SIZE = SPEED_OF_LIGHT / END_FREQUENCY
NR_STATIONS = args.nr_stations
NR_BASELINES = NR_STATIONS * (NR_STATIONS - 1) // 2

uvw = get_uvw(
    observation_hours=OBSERVATION_HOURS,
    nr_baselines=NR_BASELINES,
    grid_size=GRID_SIZE,
)

print("Initialize frequencies")
frequencies = get_frequencies(START_FREQUENCY, FREQUENCY_INCREMENT, NR_CHANNELS)
wavenumbers = (frequencies * 2 * np.pi) / SPEED_OF_LIGHT

print("Initialize metadata")
metadata = get_metadata(
    nr_channels=NR_CHANNELS,
    subgrid_size=SUBGRID_SIZE,
    grid_size=GRID_SIZE,
    uvw=uvw,
)
nr_subgrids = metadata.shape[0]


print("Parameters:")
print(f"\tnr_correlations_in: {NR_CORRELATIONS_IN}")
print(f"\tnr_correlations_out: {NR_CORRELATIONS_OUT}")
print(f"\tstart_frequency: {START_FREQUENCY*1e-6} MHz")
print(f"\tfrequency_increment: {FREQUENCY_INCREMENT*1e-6} MHz")
print(f"\tnr_channels: {NR_CHANNELS}")
print(f"\tnr_timesteps: {NR_TIMESTEPS}")
print(f"\tnr_stations: {NR_STATIONS}")
print(f"\tnr_baselines: {NR_BASELINES}")
print(f"\tsubgrid_size: {SUBGRID_SIZE}")
print(f"\tnr_subgrids: {nr_subgrids}")
print(f"\tgrid_size: {GRID_SIZE}")

print("Initialize visibilities")
start = time.time()
visibilities = get_visibilities(
    nr_correlations=NR_CORRELATIONS_IN,
    nr_channels=NR_CHANNELS,
    nr_timesteps=NR_TIMESTEPS,
    nr_baselines=NR_BASELINES,
    image_size=IMAGE_SIZE,
    grid_size=GRID_SIZE,
    frequencies=frequencies,
    uvw=uvw,
)
end = time.time()
print(f"runtime: {end-start:.2f} seconds")
print("Initialize grid")
grid = np.zeros((NR_CORRELATIONS_OUT, GRID_SIZE, GRID_SIZE), dtype=idgtypes.gridtype)

print("Initialize taper")
taper = get_taper(subgrid_size=SUBGRID_SIZE)

print("Initialize subgrids")
subgrids = np.zeros(
    shape=(nr_subgrids, NR_CORRELATIONS_OUT, SUBGRID_SIZE, SUBGRID_SIZE),
    dtype=idgtypes.gridtype,
)

print("Initialize gridder")
gridder = Gridder(
    nr_correlations_in=NR_CORRELATIONS_IN,
    subgrid_size=SUBGRID_SIZE,
)

print("Grid visibilities onto subgrids")
start = time.time()
gridder.grid_onto_subgrids(
    w_step=W_STEP,
    image_size=IMAGE_SIZE,
    grid_size=GRID_SIZE,
    wavenumbers=wavenumbers,
    uvw=uvw,
    visibilities=visibilities,
    taper=taper,
    metadata=metadata,
    subgrids=subgrids,
)
end = time.time()
print(f"runtime: {end-start:.2f} seconds")

print("Add subgrids to grid")
start = time.time()
gridder.add_subgrids_to_grid(
    metadata=metadata,
    subgrids=subgrids,
    grid=grid,
)
end = time.time()
print(f"runtime: {end-start:.2f} seconds")

print("Transform to image domain")
start = time.time()
gridder.transform(
    direction=idgtypes.FOURIER_DOMAIN_TO_IMAGE_DOMAIN,
    grid=grid,
)
end = time.time()
print(f"runtime: {end-start:.2f} seconds")

if args.store:
    print("Storing data")
    np.save("uvw.npy", uvw)
    np.save("frequencies.npy", frequencies)
    np.save("taper.npy", taper)
    np.save("metadata.npy", metadata)
    np.save("visibilities.npy", visibilities)
    np.save("subgrids.npy", subgrids)
    np.save("grid.npy", grid)
    np.save("image.npy", grid)
