module IDG

include("Input.jl")
include("Constants.jl")
include("Gridder.jl")
include("Util.jl")

function do_thing()
    inputs = Input.load_inputs()

    subgrids = zeros(ComplexF32, inputs.subgrid_count, Constants.CORRELATION_COUNT_OUT, Constants.SUBGRID_SIZE, Constants.SUBGRID_SIZE)
    grid = zeros(ComplexF32, Constants.CORRELATION_COUNT_OUT, Constants.SUBGRID_SIZE, Constants.SUBGRID_SIZE)

    Util.time_function("grid visibilities", () -> Gridder.grid_onto_subgrids!(inputs, subgrids))

    Util.time_function("ifft subgrids", () -> Gridder.ifft_subgrids!(subgrids))

    subgrids
end

end # module IDG
