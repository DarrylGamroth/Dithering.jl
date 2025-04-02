module Dithering

export Dither, target!, target, current, output, lo!, lo, hi!, hi, step_size!, step_size, reset!, step!

mutable struct Dither{T<:Number,A<:AbstractVector{T}}
    target::A
    current::A
    output::A
    error::A
    const num_actuators::Int64
    const lsb::T
    const invlsb::T
    lo::T
    hi::T
    step_size::T

    function Dither{T,A}(target, current, output, error, num_actuators, lsb, lo, hi, step) where {T<:Number,A<:AbstractVector{T}}
        if length(target) == length(current) == length(output) == length(error) != num_actuators
            throw(ArgumentError("array lengths must be equal"))
        end
        new(target, current, output, error, num_actuators, lsb, 1/lsb, lo, hi, step)
    end
end

function Dither(::Type{T}, num_actuators, bits::Int, lo, hi, step) where {T<:Number}
    return Dither{T,Vector{T}}(zeros(T, num_actuators),
        zeros(T, num_actuators),
        zeros(T, num_actuators),
        zeros(T, num_actuators),
        num_actuators, 2 / (2^bits - 1), lo, hi, step)
end

target!(d::Dither, value) = copy!(d.target, value)
target(d::Dither) = d.target
current(d::Dither) = d.current
output(d::Dither) = d.output
lo!(d::Dither, value) = d.lo = value
lo(d::Dither) = d.lo
hi!(d::Dither, value) = d.hi = value
hi(d::Dither) = d.hi
step_size!(d::Dither, value) = d.step_size = value
step_size(d::Dither) = d.step_size

function reset!(d::Dither{T,A}) where {T<:Number,A<:AbstractVector{T}}
    fill!(d.target, zero(T))
    fill!(d.current, zero(T))
    fill!(d.output, zero(T))
    fill!(d.error, zero(T))
end

function step!(d::Dither{T,A}) where {T<:Number,A<:AbstractVector{T}}
    lo = d.lo
    hi = d.hi
    lsb = d.lsb
    invlsb = d.invlsb
    step_max = d.step_size

    @inbounds for i in eachindex(d.target, d.current, d.error, d.output)
        # Load state
        x = d.target[i]
        u = d.current[i]
        e = d.error[i]

        # Ramp to target value
        delta = x - u
        u += clamp(delta, -step_max, step_max)

        # Generate triangular dither
        dither = (rand(T) - rand(T)) * lsb

        # Noise shaping filter with dither injection
        # y[i] = u[i] - e[i] + d
        y = u - e + dither

        # Quantize y
        v = round(y * invlsb) * lsb

        # Calculate quantization error
        e = v - y

        # Store state
        d.current[i] = u
        d.error[i] = e
        d.output[i] = clamp(v, lo, hi)
    end
    return d.output
end

end # module Dithering

# # This version is slower due to the use of mod()
# function update_mod!(d::Dither{T,A}) where {T<:Number,A<:AbstractVector{T}}
#     lo = d.lo
#     hi = d.hi
#     lsb = d.lsb
#     step_max = d.step

#     @inbounds for i in 1:d.length
#         x = d.target[i]
#         u = d.current[i]
#         e = d.error[i]

#         # Ramp to target value
#         delta = x - u
#         u += clamp(delta, -step_max, step_max)

#         dither = (rand(T) - 0.5) * lsb

#         # Noise shaping filter with dither injection
#         # y[i] = u[i] - e[i] + d
#         y = u - e + dither

#         # Calculate quantization error
#         e = -mod(y, lsb)

#         # Calculate quantized output
#         v = y + e

#         d.current[i] = u
#         d.error[i] = e
#         d.output[i] = clamp(v, lo, hi)
#     end
#     return d.output
# end
