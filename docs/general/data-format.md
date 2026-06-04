In order to test the various applications with identical inputs and to be able to compare the outputs, it is important that all applications write their output to a single file format.

At first, the numpy library's NPY format was chosen as the standard data format, as there were NPY libraries for each language in the project and it supported our usecases. However, after working with some of these libraries it became obvious that support was not adequate. Specifically, beside the Python library none of the NPY libraries completely supported compound datatypes.  At this point a reevaluation was decided on.

## Evaluating data-formats

For the evaluation of different data formats, the following formats were considered:

1. numpy's NPY/NPZ
2. HDF5
3. Parqueet
4. Safetensors
5. Apache Avro

The following factors were considered:

1. Supported features
    a. Complex Numbers
    b. Compound Datatypes
    c. N-dimensional Arrays
2. Library support
3. Ease of use

### Evaluation matrix

The following evaluation was made:

| Format      | Supported Features | Library support | Ease of use |
| :---------- | -----------------: | --------------: | ----------: |
| NPY         |               Good |            Poor |        Good |
| HDF5        |               Good |            Good |        Good |
| Parqueet    |                Bad |               / |           / |
| Safetensors |                Bad |               / |           / |
| Avro        |                Bad |               / |           / |

In this evaluation HDF5 was the clear favorite. Since it is also commonly used in science and specifically astronomy, HDF5 was picked.

### Library selection

To read/write HDF5, the following libraries were chosen:

| Language | Library                                              |
| -------- | ---------------------------------------------------- |
| Python   | [h5py](https://docs.h5py.org/en/stable/index.html)   |
| C++      | [h5cpp](https://github.com/ess-dmsc/h5cpp)           |
| Rust     | [hdf5-metno](https://crates.io/crates/hdf5-metno)    |
| Julia    | [HDF5.jl](https://juliaio.github.io/HDF5.jl/stable/) |
