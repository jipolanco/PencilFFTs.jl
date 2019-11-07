## Real-to-complex and complex-to-real transforms.

"""
    RFFT()

Real-to-complex FFT.

See also
[`AbstractFFTs.rfft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.rfft).
"""
struct RFFT <: AbstractTransform end

"""
    IRFFT()

Normalised inverse of [`RFFT`](@ref).

See also
[`AbstractFFTs.irfft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.irfft).
"""
struct IRFFT <: AbstractTransform end

"""
    BRFFT()

Unnormalised inverse of [`RFFT`](@ref).

To obtain the inverse transform, divide the output by the length of the
transformed dimension (of the real output array).

See also
[`AbstractFFTs.brfft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.brfft).
"""
struct BRFFT <: AbstractTransform end

const TransformR2C = RFFT
const TransformC2R = Union{IRFFT, BRFFT}

length_output(::TransformR2C, length_in::Integer) = div(length_in, 2) + 1
length_output(::TransformC2R, length_in::Integer) = 2 * length_in - 2

eltype_output(::TransformR2C, ::Type{T}) where {T <: FFTReal} = Complex{T}
eltype_output(::TransformC2R, ::Type{Complex{T}}) where {T <: FFTReal} = T

eltype_input(::TransformR2C, ::Type{T}) where {T <: FFTReal} = T
eltype_input(::TransformC2R, ::Type{T}) where {T <: FFTReal} = Complex{T}

# Backward plans: we assume that the input data size is even!
_args_bw_rfft((A, dims)) = (A, 2 * size(A, first(dims)) - 2, dims)

plan(::RFFT, args...; kwargs...) = FFTW.plan_rfft(args...; kwargs...)
plan(::IRFFT, args...; kwargs...) =
    FFTW.plan_irfft(_args_bw_rfft(args)...; kwargs...)
plan(::BRFFT, args...; kwargs...) =
    FFTW.plan_brfft(_args_bw_rfft(args)...; kwargs...)

inv(::TransformR2C) = BRFFT()
inv(::TransformC2R) = RFFT()

# r2c along the first dimension, then c2c for the other dimensions.
expand_dims(::RFFT, ::Val{N}) where N =
    N === 0 ? () : (RFFT(), expand_dims(FFT(), Val(N - 1))...)
expand_dims(::IRFFT, ::Val{N}) where N =
    N === 0 ? () : (IRFFT(), expand_dims(IFFT(), Val(N - 1))...)
expand_dims(::BRFFT, ::Val{N}) where N =
    N === 0 ? () : (BRFFT(), expand_dims(BFFT(), Val(N - 1))...)
