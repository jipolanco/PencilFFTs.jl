## Real-to-complex and complex-to-real transforms.

"""
    RFFT()

Real-to-complex FFT.

See also
[`AbstractFFTs.rfft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.rfft).
"""
struct RFFT <: AbstractTransform end

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
const TransformC2R = BRFFT

length_output(::TransformR2C, length_in::Integer) = div(length_in, 2) + 1
length_output(::TransformC2R, length_in::Integer) = 2 * length_in - 2

eltype_output(::TransformR2C, ::Type{T}) where {T <: FFTReal} = Complex{T}
eltype_output(::TransformC2R, ::Type{Complex{T}}) where {T <: FFTReal} = T

eltype_input(::TransformR2C, ::Type{T}) where {T <: FFTReal} = T
eltype_input(::TransformC2R, ::Type{T}) where {T <: FFTReal} = Complex{T}

# Backward plans: in the first case, we assume that the input data size is even!
_args_bw_rfft(A, dims) = (A, 2 * size(A, first(dims)) - 2, dims)
_args_bw_rfft(A, d, dims) = (A, d, dims)

plan(::RFFT, args...; kwargs...) = FFTW.plan_rfft(args...; kwargs...)
plan(::BRFFT, args...; kwargs...) =
    FFTW.plan_brfft(_args_bw_rfft(args...)...; kwargs...)

binv(::RFFT) = BRFFT()
binv(::BRFFT) = RFFT()

# Note: the output of RFFT (BRFFT) is complex (real).
scale_factor(::BRFFT, A::RealArray, dims) = _prod_dims(A, dims)
scale_factor(::RFFT, A::ComplexArray, dims) = _scale_factor_r2c(A, dims...)

function _scale_factor_r2c(A::ComplexArray, dim1, dims...)
    # I need to normalise by the *logical* size of the real output.
    # We assume that the dimension `dim1` is the dimension with Hermitian
    # symmetry.
    s = size(A)
    2 * (s[dim1] - 1) * _intprod((s[i] for i in dims)...)
end

# r2c along the first dimension, then c2c for the other dimensions.
expand_dims(::RFFT, ::Val{N}) where N =
    N === 0 ? () : (RFFT(), expand_dims(FFT(), Val(N - 1))...)
expand_dims(::BRFFT, ::Val{N}) where N =
    N === 0 ? () : (BRFFT(), expand_dims(BFFT(), Val(N - 1))...)
