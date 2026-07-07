# IDG Input Generator

This tool generates input that can be consumed by an IDG implementation.

## Prerequisites

To run this Python program you will need to install the UV package manager. It can be installed from <https://docs.astral.sh/uv/getting-started/installation/>, or if you already have `pip` you can install it with:

```bash
pip install uv
```

## Usage

In `idg/input` (this folder), execute `uv run idg-input`. This will create a file named `inputs.h5` in the same directory.  
There are command-line options available to customize the parameters used for the generation. You can view these with `uv run idg-input --help`.

## Output
The output is a single HDF5 file containing the following datasets:

- `uvws`: UVW coordinates of each baseline over time. Shape `(nr_baselines, nr_timesteps)`.
- `frequencies`: Channel frequencies in Hertz. Shape `(nr_channels)`.
- `metadata`: Various metadata for each subgrid. Shape `(nr_subgrids)`.
- `visibilities`: Visibilities. Shape `(nr_baselines, nr_time, nr_channels, nr_correlations)`.

The file also contains the `grid_size` and `subgrid_size` that the input was generated with as attributes.

HDF5 files can be inspected using [`h5ls`](https://portal.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html), [HDFView](https://www.hdfgroup.org/download-hdfview/), or [HDF Compass](https://github.com/HDFGroup/hdf-compass).
