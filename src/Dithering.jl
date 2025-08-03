module Dithering

export Dither, target!, target, current, output, lo!, lo, hi!, hi, step_size!, step_size, reset!, step!

mutable struct Dither{T<:Number,A<:AbstractVector{T}}
    const target::A
    const current::A
    const output::A
    const error::A
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
        new(target, current, output, error, num_actuators, lsb, 1 / lsb, lo, hi, step)
    end
end

"""
    Dither(::Type{T}, num_actuators, bits::Int, lo, hi, step) where {T<:Number}

Constructs a `Dither` object with the specified parameters.

# Arguments
- `::Type{T}`: The numeric type of the dither values (e.g., `Float64`, `Int`).
- `num_actuators::Int`: The number of actuators to be used.
- `bits::Int`: The number of bits used for quantization.
- `lo`: The lower bound of the dither range.
- `hi`: The upper bound of the dither range.
- `step`: The step size to be used for ramping to the target value.

# Returns
A `Dither{T, Vector{T}}` object initialized with zeroed vectors and the specified parameters.

# Notes
The quantization step size is calculated as `2 / (2^bits - 1)`.
"""
function Dither(::Type{T}, num_actuators, bits::Int, lo, hi, step) where {T<:Number}
    return Dither{T,Vector{T}}(zeros(T, num_actuators),
        zeros(T, num_actuators),
        zeros(T, num_actuators),
        zeros(T, num_actuators),
        num_actuators, 2 / (2^bits - 1), lo, hi, step)
end

"""
    target!(d::Dither, value)

Sets the target value array
"""
target!(d::Dither{T}, value::AbstractVector{T}) where {T} = copyto!(d.target, value)
target(d::Dither) = d.target
current(d::Dither) = d.current
output(d::Dither) = d.output
lo!(d::Dither, value) = d.lo = value
lo(d::Dither) = d.lo
hi!(d::Dither, value) = d.hi = value
hi(d::Dither) = d.hi
step_size!(d::Dither, value) = d.step_size = value
step_size(d::Dither) = d.step_size

"""
    reset!(d::Dither)

    Reset the state
"""
function reset!(d::Dither{T}) where {T}
    fill!(d.target, zero(T))
    fill!(d.current, zero(T))
    fill!(d.output, zero(T))
    fill!(d.error, zero(T))
end

"""
    step!(d::Dither)

    Perform a single step of the dithering process.
"""
function step!(d::Dither{T}) where {T}
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

        # Store state
        d.current[i] = u
        d.error[i] = v - y
        d.output[i] = clamp(v, lo, hi)
    end
    return d.output
end

end # module Dithering
