module Gridder

include("Types.jl")
include("Input.jl")
include("Constants.jl")
include("Util.jl")

function grid_onto_subgrids!(inputs, subgrids)
    Threads.@threads for i in 1:length(inputs.metadata)
        visibility_to_subgrid!(
            inputs.metadata[i],
            inputs.w_step,
            inputs.image_size,
            Constants.GRID_SIZE,
            inputs.correlation_count_in,
            inputs.correlation_count_out,
            inputs.wavenumbers,
            inputs.uvws,
            inputs.visibilities,
            inputs.taper,
            @view subgrids[i, :, :, :]
        )
    end
end

function visibility_to_subgrid!(
    metadatum,
    w_step,
    image_size,
    grid_size,
    correlation_count_in,
    correlation_count_out,
    wavenumbers,
    uvw,
    visibilities,
    taper,
    subgrid,
)
    w_offset_in_lambda = w_step * (metadatum.coordinate.z + 0.5)

    u_offset = (metadatum.coordinate.x + Constants.SUBGRID_SIZE / 2 - Constants.GRID_SIZE / 2) * (2 * π / image_size)
    v_offset = (metadatum.coordinate.y + Constants.SUBGRID_SIZE / 2 - Constants.GRID_SIZE / 2) * (2 * π / image_size)
    w_offset = 2 * π * w_offset_in_lambda

    for y in (1:Constants.SUBGRID_SIZE)
        for x in (1:Constants.SUBGRID_SIZE)
            l = compute_lm(x - 1, Constants.SUBGRID_SIZE, image_size)
            m = compute_lm(y - 1, Constants.SUBGRID_SIZE, image_size)
            n = compute_n(l, m)

            pixels = compute_pixels(
                metadatum.timestep_count,
                metadatum.time_index,
                uvw,
                metadatum.baseline,
                l,
                m,
                n,
                u_offset,
                v_offset,
                w_offset,
                metadatum.channel_begin,
                metadatum.channel_end,
                correlation_count_in,
                correlation_count_out,
                wavenumbers,
                visibilities
            )

            sph = taper[y, x]
            x_dst = Util.array_mod((x - 1 + (Constants.SUBGRID_SIZE ÷ 2)), Constants.SUBGRID_SIZE)
            y_dst = Util.array_mod((y - 1 + (Constants.SUBGRID_SIZE ÷ 2)), Constants.SUBGRID_SIZE)

            for pol in 1:correlation_count_out
                subgrid[pol, y_dst, x_dst] = pixels[pol] * sph
            end
        end
    end
end


compute_lm(x, subgrid_size, image_size) = (x + 0.5 - (subgrid_size / 2.0)) * image_size / subgrid_size
function compute_n(l, m)
    tmp = l * l + m * m

    if tmp >= 1.0
        return 1.0
    end

    tmp / (1.0 + √(1 - tmp))
end

function compute_pixels(
    timestep_count,
    offset,
    uvws,
    baseline,
    l,
    m,
    n,
    u_offset,
    v_offset,
    w_offset,
    channel_begin,
    channel_end,
    correlation_count_in,
    correlation_count_out,
    wavenumbers,
    visibilities
)
    pixels = zeros(ComplexF32, Constants.CORRELATION_COUNT_OUT)

    for time in 0:timestep_count-1
        idx = offset + time
        (; u, v, w) = uvws[baseline+1, idx+1]

        phase_index = u * l + v * m + w * n
        phase_offset = u_offset * l + v_offset * m + w_offset * n

        for channel in channel_begin+1:channel_end
            phase = phase_offset - (phase_index * wavenumbers[channel])
            phasor = exp(1im * phase)

            for pol in 1:correlation_count_in
                pixels[Util.array_mod(pol, Constants.CORRELATION_COUNT_OUT)] += visibilities[baseline+1, idx+1, channel, pol] * phasor
            end
        end
    end

    pixels
end

end
