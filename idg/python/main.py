import numpy as np
import time

import idgtypes
from idg import Gridder  # type: ignore

from init import (
    get_uvw,
    get_metadata,
    get_frequencies,
    get_visibilities,
    get_taper,
)

nr_correlations_in = 2  # XX, YY
nr_correlations_out = 1  # I
subgrid_size = 32  # size of each subgrid
grid_size = 1024  # size of the full grid
observation_hours = 4  # total observation time in hours
nr_timesteps = observation_hours * 3600
nr_channels = 16  # number of frequency channels
w_step = 1.0  # w step in wavelengths

start_frequency = 150e6  # 150 MHz
frequency_increment = 1e6  # 1 MHz
end_frequency = start_frequency + nr_channels * frequency_increment

speed_of_light = 299792458.0
image_size = speed_of_light / end_frequency

nr_stations = 20
nr_baselines = nr_stations * (nr_stations - 1) // 2

uvw = get_uvw(
    observation_hours=observation_hours,
    nr_baselines=nr_baselines,
    grid_size=grid_size,
)
np.save("uvw.npy", uvw)

print("Initialize frequencies")
frequencies = get_frequencies(start_frequency, frequency_increment, nr_channels)
wavenumbers = (frequencies * 2 * np.pi) / speed_of_light
np.save("frequencies.npy", frequencies)

print("Initialize metadata")
metadata = get_metadata(
    nr_channels=nr_channels,
    subgrid_size=subgrid_size,
    grid_size=grid_size,
    uvw=uvw,
)
nr_subgrids = metadata.shape[0]
np.save("metadata.npy", metadata)


print(f"Parameters:")
print(f"\tnr_correlations_in: {nr_correlations_in}")
print(f"\tnr_correlations_out: {nr_correlations_out}")
print(f"\tstart_frequency: {start_frequency*1e-6} MHz")
print(f"\tfrequency_increment: {frequency_increment*1e-6} MHz")
print(f"\tnr_channels: {nr_channels}")
print(f"\tnr_timesteps: {nr_timesteps}")
print(f"\tnr_stations: {nr_stations}")
print(f"\tnr_baselines: {nr_baselines}")
print(f"\tsubgrid_size: {subgrid_size}")
print(f"\tnr_subgrids: {nr_subgrids}")
print(f"\tgrid_size: {grid_size}")

print("Initialize visibilities")
start = time.time()
visibilities = get_visibilities(
    nr_correlations=nr_correlations_in,
    nr_channels=nr_channels,
    nr_timesteps=nr_timesteps,
    nr_baselines=nr_baselines,
    image_size=image_size,
    grid_size=grid_size,
    frequencies=frequencies,
    uvw=uvw,
)
end = time.time()
print(f"runtime: {end-start:.2f} seconds")
np.save("visibilities.npy", visibilities)

print("Initialize grid")
grid = np.zeros((nr_correlations_out, grid_size, grid_size), dtype=idgtypes.gridtype)

print("Initialize taper")
taper = get_taper(subgrid_size=subgrid_size)
np.save("taper.npy", taper)

# allocate subgrids
subgrids = np.zeros(
    shape=(nr_subgrids, nr_correlations_out, subgrid_size, subgrid_size),
    dtype=idgtypes.gridtype,
)

print("Initialize gridder")
gridder = Gridder(
    nr_correlations_in=nr_correlations_in,
    subgrid_size=subgrid_size,
)

print("Grid visibilities onto subgrids")
start = time.time()
gridder.grid_onto_subgrids(
    w_step=w_step,
    image_size=image_size,
    grid_size=grid_size,
    wavenumbers=wavenumbers,
    uvw=uvw,
    visibilities=visibilities,
    taper=taper,
    metadata=metadata,
    subgrids=subgrids,
)
end = time.time()
print(f"runtime: {end-start:.2f} seconds")
np.save("subgrids.npy", subgrids)

print("Add subgrids to grid")
start = time.time()
gridder.add_subgrids_to_grid(
    metadata=metadata,
    subgrids=subgrids,
    grid=grid,
)
end = time.time()
print(f"runtime: {end-start:.2f} seconds")

print("Save grid to grid1.npy")
np.save("grid1.npy", grid)

print("Transform to image domain")
start = time.time()
gridder.transform(
    direction=idgtypes.FourierDomainToImageDomain,
    grid=grid,
)
end = time.time()
print(f"runtime: {end-start:.2f} seconds")

print("Save grid to grid2.npy")
np.save("grid2.npy", grid)
