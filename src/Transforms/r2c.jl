## Real-to-complex and complex-to-real transforms.

"""
    RFFT()

Real-to-complex FFT.

See also
[`AbstractFFTs.rfft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.rfft).
"""
struct RFFT <: AbstractTransform end

"""
    BRFFT(d::Integer)
    BRFFT((d1, d2, ..., dN))

Unnormalised inverse of [`RFFT`](@ref).

To obtain the inverse transform, divide the output by the length of the
transformed dimension (of the real output array).

As described in the [AbstractFFTs docs](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.irfft),
the length of the output cannot be fully inferred from the input length.
For this reason, the `BRFFT` constructor accepts an optional `d` argument
indicating the output length.

For multidimensional datasets, a tuple of dimensions
`(d1, d2, ..., dN)` may also be passed.
This is equivalent to passing just `dN`.
In this case, the **last** dimension (`dN`) is the one that changes size between
the input and output.
Note that this is the opposite of `FFTW.brfft`.
The reason is that, in PencilFFTs, the **last** dimension is the one along which
a complex-to-real transform is performed.

See also
[`AbstractFFTs.brfft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.brfft).
"""
struct BRFFT <: AbstractTransform
    even_output :: Bool
end

_show_extra_info(io::IO, tr::BRFFT) = print(io, tr.even_output ? "{even}" : "{odd}")

BRFFT(d::Integer) = BRFFT(iseven(d))
BRFFT(ts::Tuple) = BRFFT(last(ts))  # c2r transform is applied along the **last** dimension (opposite of FFTW)

is_inplace(::Union{RFFT, BRFFT}) = false

length_output(::RFFT, length_in::Integer) = div(length_in, 2) + 1
length_output(tr::BRFFT, length_in::Integer) = 2 * length_in - 1 - tr.even_output

eltype_output(::RFFT, ::Type{T}) where {T <: FFTReal} = Complex{T}
eltype_output(::BRFFT, ::Type{Complex{T}}) where {T <: FFTReal} = T

eltype_input(::RFFT, ::Type{T}) where {T <: FFTReal} = T
eltype_input(::BRFFT, ::Type{T}) where {T <: FFTReal} = Complex{T}

# NOTE: unlike most FFTW plans, brfft function also requires the length `d` of
# the transform output along the first transformed dimension.
function plan(tr::BRFFT, A::AbstractArray, dims; kwargs...)
    Nin = size(A, first(dims))  # input length along first dimension
    d = length_output(tr, Nin)
    plan_brfft(A, d, dims)
end
function plan(tr::BRFFT, A::Array, dims; kwargs...)
    Nin = size(A, first(dims))  # input length along first dimension
    d = length_output(tr, Nin)
    FFTW.plan_brfft(A, d, dims; kwargs...)
end
plan(::RFFT,  args...; kwargs...) = plan_rfft(args...)
plan(::RFFT,  X::Array, args...; kwargs...) = FFTW.plan_rfft(X,  args...; kwargs...)

binv(::RFFT)  = BRFFT()
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
>>>>>>> 09cfbd0 (Changes in order to take into account AbstractVectors instead of just (CPU) Vectors)
end

binv(::RFFT, d) = BRFFT(d)
binv(::BRFFT, d) = RFFT()

function scale_factor(tr::BRFFT, A::ComplexArray, dims)
    prod(dims; init = one(Int)) do i
        n = size(A, i)
        i == last(dims) ? length_output(tr, n) : n
    end
end

scale_factor(::RFFT, A::RealArray, dims) = _prod_dims(A, dims)

# r2c along the first dimension, then c2c for the other dimensions.
expand_dims(tr::RFFT, ::Val{N}) where {N} =
    N === 0 ? () : (tr, expand_dims(FFT(), Val(N - 1))...)

expand_dims(tr::BRFFT, ::Val{N}) where {N} = (BFFT(), expand_dims(tr, Val(N - 1))...)
expand_dims(tr::BRFFT, ::Val{1}) = (tr, )
expand_dims(tr::BRFFT, ::Val{0}) = ()
