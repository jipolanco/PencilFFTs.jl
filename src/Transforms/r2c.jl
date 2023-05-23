## Real-to-complex and complex-to-real transforms.

"""
    RFFT()

Real-to-complex FFT.

See also
[`AbstractFFTs.rfft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.rfft).
"""
struct RFFT <: AbstractTransform end

"""
    RFFT!()

In-place version of [`RFFT`](@ref).
"""
struct RFFT! <: AbstractTransform end

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

"""
    BRFFT!(d::Integer)
    BRFFT!((d1, d2, ..., dN))

In-place version of [`BRFFT`](@ref).
"""
struct BRFFT! <: AbstractTransform
    even_output :: Bool
end

const TransformR2C = Union{RFFT, RFFT!}
const TransformC2R = Union{BRFFT, BRFFT!}

_show_extra_info(io::IO, tr::TransformC2R) = print(io, tr.even_output ? "{even}" : "{odd}")

BRFFT(d::Integer) = BRFFT(iseven(d))
BRFFT(ts::Tuple) = BRFFT(last(ts))  # c2r transform is applied along the **last** dimension (opposite of FFTW)
BRFFT!(d::Integer) = BRFFT!(iseven(d))
BRFFT!(ts::Tuple) = BRFFT!(last(ts))  # c2r transform is applied along the **last** dimension (opposite of FFTW)

is_inplace(::Union{RFFT, BRFFT}) = false
is_inplace(::Union{RFFT!, BRFFT!}) = true

length_output(::TransformR2C, length_in::Integer) = div(length_in, 2) + 1
length_output(tr::TransformC2R, length_in::Integer) = 2 * length_in - 1 - tr.even_output

eltype_output(::TransformR2C, ::Type{T}) where {T <: FFTReal} = Complex{T}
eltype_output(::TransformC2R, ::Type{Complex{T}}) where {T <: FFTReal} = T

eltype_input(::TransformR2C, ::Type{T}) where {T <: FFTReal} = T
eltype_input(::TransformC2R, ::Type{T}) where {T <: FFTReal} = Complex{T}

plan(::RFFT, A::AbstractArray, args...; kwargs...) = FFTW.plan_rfft(A, args...; kwargs...)
plan(::RFFT!, A::AbstractArray, args...; kwargs...) = plan_rfft!(A, args...; kwargs...)

# NOTE: unlike most FFTW plans, this function also requires the length `d` of
# the transform output along the first transformed dimension.
function plan(tr::BRFFT, A::AbstractArray, dims; kwargs...)
    Nin = size(A, first(dims))  # input length along first dimension
    d = length_output(tr, Nin)
    FFTW.plan_brfft(A, d, dims; kwargs...)
end
function plan(tr::BRFFT!, A::AbstractArray, dims; kwargs...)
    Nin = size(A, first(dims))  # input length along first dimension
    d = length_output(tr, Nin)
    plan_brfft!(A, d, dims; kwargs...) 
end

binv(::RFFT, d) = BRFFT(d)
binv(::BRFFT, d) = RFFT()
binv(::RFFT!, d) = BRFFT!(d)
binv(::BRFFT!, d) = RFFT!()

function scale_factor(tr::TransformC2R, A::ComplexArray, dims)
    prod(dims; init = one(Int)) do i
        n = size(A, i)
        i == last(dims) ? length_output(tr, n) : n
    end
end

scale_factor(::TransformR2C, A::RealArray, dims) = _prod_dims(A, dims)

# r2c along the first dimension, then c2c for the other dimensions.
expand_dims(tr::RFFT, ::Val{N}) where {N} =
    N === 0 ? () : (tr, expand_dims(FFT(), Val(N - 1))...)
expand_dims(tr::RFFT!, ::Val{N}) where {N} =
    N === 0 ? () : (tr, expand_dims(FFT!(), Val(N - 1))...)

expand_dims(tr::BRFFT, ::Val{N}) where {N} = (BFFT(), expand_dims(tr, Val(N - 1))...)
expand_dims(tr::BRFFT!, ::Val{N}) where {N} = (BFFT!(), expand_dims(tr, Val(N - 1))...)
expand_dims(tr::TransformC2R, ::Val{1}) = (tr, )
expand_dims(tr::TransformC2R, ::Val{0}) = ()

## FFTW wrappers for inplace RFFT plans

function plan_rfft!(X::StridedArray{T,N}, region;
    flags::Integer=FFTW.ESTIMATE,
    timelimit::Real=FFTW.NO_TIMELIMIT) where {T<:FFTW.fftwReal,N}
    sz = size(X) # physical input size (real)
    osize = FFTW.rfft_output_size(sz, region) # output size (complex)
    isize = ntuple(i -> i == first(region) ? 2osize[i] : osize[i], Val(N)) # padded input size (real)
    X_padded = flags&FFTW.ESTIMATE != 0 ? # is time measurement not required ?
        FFTW.FakeArray{T,N}(sz, FFTW.colmajorstrides(isize)) : # fake allocation, only size and strides matter
        view(Array{T}(undef, isize), Base.OneTo.(sz)...) # allocation
    Y = flags&FFTW.ESTIMATE != 0 ? 
        FFTW.FakeArray{Complex{T}}(osize) : # fake allocation, only size and strides matter
        Array{Complex{T}}(undef, osize) # allocation
    return FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N}(X_padded, Y, region, flags, timelimit)
end

function plan_brfft!(X::StridedArray{Complex{T},N}, d, region;
    flags::Integer=FFTW.ESTIMATE,
    timelimit::Real=FFTW.NO_TIMELIMIT) where {T<:FFTW.fftwReal,N}
    isize = size(X) # input size (complex)
    osize = ntuple(i -> i == first(region) ? 2isize[i] : isize[i], Val(N)) # padded output size (real)
    sz = FFTW.brfft_output_size(X, d, region) # physical output size (real)
    # Y is padded
    Y = flags&FFTW.ESTIMATE != 0 ?  # is time measurement not required ?
        FFTW.FakeArray{T,N}(sz, FFTW.colmajorstrides(osize)) : # fake allocation, only size and strides matter
        view(Array{T}(undef, osize), Base.OneTo.(sz)...) # allocation
    return FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,true,N}(X, Y, region, flags, timelimit)
end
#= function Base.:*(p::FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N}, X::StridedArray{T,N}) where {T<:FFTW.fftwReal,N}
    ptr_complex = convert(Ptr{Complex{T}}, pointer(X)) 
    Y = unsafe_wrap(Array, ptr_complex, p.osz) # reinterpretation of X as a complex Array
    mul!(Y, p, X) # assesses size, strides and memory alignment
end

function Base.:*(p::FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,true,N}, X::StridedArray{Complex{T},N}) where {T<:FFTW.fftwReal,N}
    ptr_real = convert(Ptr{T}, pointer(X))
    osize = ntuple(i -> i == first(p.region) ? 2p.sz[i] : p.sz[i], Val(N)) # padded real size
    Y_parent = unsafe_wrap(Array, ptr_real, osize) # reinterpretation of X as a real Array
    Y = view(Y_parent, Base.OneTo.(p.osz)...) # skip padded values, view to physical real size only
    mul!(Y, p, X) # assesses size, strides and memory alignment
end =#