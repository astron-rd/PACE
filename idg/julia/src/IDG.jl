module IDG

include("Input.jl")
include("Constants.jl")
include("Gridder.jl")
include("Util.jl")

using HDF5

function run()
    inputs = Input.load_inputs()

    subgrids = zeros(ComplexF32, inputs.subgrid_count, Constants.CORRELATION_COUNT_OUT, Constants.SUBGRID_SIZE, Constants.SUBGRID_SIZE)
    grid = zeros(ComplexF32, Constants.CORRELATION_COUNT_OUT, Constants.GRID_SIZE, Constants.GRID_SIZE)

    Util.time_function("grid visibilities", () -> Gridder.grid_onto_subgrids!(inputs, subgrids))

    Util.time_function("ifft subgrids", () -> Gridder.ifft_subgrids!(subgrids))

    Util.time_function("add subgrids to grid", () -> Gridder.add_subgrids_to_grid!(inputs, subgrids, grid))

    Util.time_function("transform grid", () -> Gridder.transform_grid!(grid))

    grid = Util.reverse_dims(grid)

    outfile = HDF5.h5open("output.h5", "w")
    create_dataset(outfile, "grid", grid)
    write(outfile["grid"], grid)
end

end # module IDG
