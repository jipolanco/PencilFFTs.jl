## Real-to-complex and complex-to-real transforms.

"""
    RFFT()

Real-to-complex FFT.

See also
[`AbstractFFTs.rfft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.rfft).
"""
struct RFFT <: AbstractTransform end

"""
    BRFFT([even_output::Bool = true])
    BRFFT(d::Integer)

Unnormalised inverse of [`RFFT`](@ref).

To obtain the inverse transform, divide the output by the length of the
transformed dimension (of the real output array).

As described in the [AbstractFFTs docs](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.irfft),
the length of the output cannot be fully inferred from the input length.
For this reason, the `BRFFT` constructor accepts an optional `d` argument
indicating the output length.
Alternatively, a `Bool` argument may be passed indicating whether the output has
even or odd length.
By default, if nothing is passed, the output is assumed to have even length.

See also
[`AbstractFFTs.brfft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.brfft).
"""
struct BRFFT <: AbstractTransform
    even_output :: Bool
end

_show_extra_info(io::IO, tr::BRFFT) = print(io, tr.even_output ? "{even}" : "{odd}")

BRFFT(d::Integer) = BRFFT(iseven(d))
BRFFT() = BRFFT(true)

is_inplace(::Union{RFFT, BRFFT}) = false

length_output(::RFFT, length_in::Integer) = div(length_in, 2) + 1
length_output(tr::BRFFT, length_in::Integer) = 2 * length_in - 1 - tr.even_output

eltype_output(::RFFT, ::Type{T}) where {T <: FFTReal} = Complex{T}
eltype_output(::BRFFT, ::Type{Complex{T}}) where {T <: FFTReal} = T

eltype_input(::RFFT, ::Type{T}) where {T <: FFTReal} = T
eltype_input(::BRFFT, ::Type{T}) where {T <: FFTReal} = Complex{T}

plan(::RFFT, args...; kwargs...) = FFTW.plan_rfft(args...; kwargs...)

# NOTE: unlike most FFTW plans, this function also requires the length `d` of
# the transform output along the first transformed dimension.
function plan(tr::BRFFT, A, dims; kwargs...)
    Nin = size(A, first(dims))  # input length along first dimension
    d = length_output(tr, Nin)
    FFTW.plan_brfft(A, d, dims; kwargs...)
end

binv(::RFFT, d) = BRFFT(d)
binv(::BRFFT, d) = RFFT()

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
