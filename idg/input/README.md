# IDG Input Generator

This tool generates input that can be consumed by an IDG implementation.

## Output
The output is a single HDF5 file containing the following datasets:

- `uvws`
- `frequencies`
- `metadata`
- `visibilities`

It also contains the `grid_size` and `subgrid_size` that the input was generated with.

## Usage

In `idg/input` (this folder), execute `uv run idg-input`.
