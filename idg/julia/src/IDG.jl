module IDG

include("Input.jl")
include("Constants.jl")
include("Gridder.jl")

function do_thing()
    inputs = Input.load_inputs()

    subgrids = zeros(ComplexF32, inputs.subgrid_count, Constants.CORRELATION_COUNT_OUT, Constants.SUBGRID_SIZE, Constants.SUBGRID_SIZE)
    grid = zeros(ComplexF32, Constants.CORRELATION_COUNT_OUT, Constants.SUBGRID_SIZE, Constants.SUBGRID_SIZE)

    Gridder.grid_onto_subgrids!(inputs, subgrids)

    subgrids
end

end # module IDG
