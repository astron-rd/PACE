module Types

UVW = @NamedTuple{u::Float32, v::Float32, w::Float32}

Metadata = @NamedTuple{baseline::UInt32, time_index::UInt32, nr_timesteps::UInt32, channel_begin::UInt32, channel_end::UInt32, coordinate::@NamedTuple{x::UInt32, y::UInt32, z::UInt32}}

function generate_taper(subgrid_size::Integer)
    x = LinRange(-1.0, 1.0, subgrid_size)
    spheroidal = evaluate_spheroidal.(x)

    mat_1n = permutedims(spheroidal)
    mat_n1 = transpose(mat_1n)

    mat_1n .* mat_n1
end

function evaluate_spheroidal(x)
    P = [
        [8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
        [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2],
    ]
    Q = [
        [1.0000000e0, 8.212018e-1, 2.078043e-1],
        [1.0000000e0, 9.599102e-1, 2.918724e-1],
    ]

    (part, endi) = if x < 0.75
        (1, 0.75)
    else
        (2, 1.0)
    end

    x_squared = x^2
    delta_x_squared = x_squared - endi^2
    top = evaluate_polynomial(delta_x_squared, P[part])
    btm = evaluate_polynomial(delta_x_squared, Q[part])

    if btm == 0.0
        0.0
    else
        (1.0 - x_squared) * (top / btm)
    end
end

function evaluate_polynomial(x, coeff)
    val = coeff[1]
    x_accumulator = x
    for p in coeff[2:end]
        val += p * x_accumulator
        x_accumulator *= x
    end
    val
end

end
