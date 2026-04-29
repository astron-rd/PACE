module Input

include("Types.jl")
include("Util.jl")
include("Constants.jl")

using .Types
using .Util
using HDF5

struct Inputs
    uvws
    frequencies
    wavenumbers
    visibilities
    metadata
    taper
    subgrid_count
    image_size
    correlation_count_in
    correlation_count_out
    w_step
end

function load_inputs()
    Util.print_header("READING INPUT DATA")
    infile = HDF5.h5open("input/input.hdf5", "r")

    uvw_ds = infile["uvws"]
    uvws = Util.time_function("load uvws", () -> read(uvw_ds))
    uvws = permutedims(uvws, reverse(1:ndims(uvws)))

    frequencies_ds = infile["frequencies"]
    frequencies = Util.time_function("load frequencies", () -> read(frequencies_ds))

    wavenumbers = Util.time_function("derive wavenumbers", () -> (frequencies .* 2.0 .* π) ./ Constants.SPEED_OF_LIGHT)

    metadata_ds = infile["metadata"]
    metadata = Util.time_function("load metadata", () -> read(metadata_ds))
    metadata = permutedims(metadata, reverse(1:ndims(metadata)))

    visibilities_ds = infile["visibilities"]
    visibilities = Util.time_function("load visibilities", () -> read(visibilities_ds))
    visibilities = permutedims(visibilities, reverse(1:ndims(visibilities)))

    taper = Types.generate_taper(Constants.SUBGRID_SIZE)

    Inputs(
        uvws,
        frequencies,
        wavenumbers,
        visibilities,
        metadata,
        taper,
        length(metadata),
        Constants.SPEED_OF_LIGHT / last(frequencies),
        size(visibilities)[4],
        Constants.CORRELATION_COUNT_OUT,
        Constants.W_STEP
    )
end

end
