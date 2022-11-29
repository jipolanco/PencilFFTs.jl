## Complex-to-complex transforms.

"""
    FFT()

Complex-to-complex FFT.

See also
[`AbstractFFTs.fft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.fft).
"""
struct FFT <: AbstractTransform end

"""
    FFT!()

In-place version of [`FFT`](@ref).

See also
[`AbstractFFTs.fft!`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.fft!).
"""
struct FFT! <: AbstractTransform end

"""
    BFFT()

Unnormalised backward complex-to-complex FFT.

Like `AbstractFFTs.bfft`, this transform is not normalised.
To obtain the inverse transform, divide the output by the length of the
transformed dimension.

See also
[`AbstractFFTs.bfft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.bfft).
"""
struct BFFT <: AbstractTransform end

"""
    BFFT()

In-place version of [`BFFT`](@ref).

See also
[`AbstractFFTs.bfft!`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.bfft!).
"""
struct BFFT! <: AbstractTransform end

const TransformC2C = Union{FFT, FFT!, BFFT, BFFT!}

length_output(::TransformC2C, length_in::Integer) = length_in
eltype_output(::TransformC2C, ::Type{Complex{T}}) where {T <: FFTReal} = Complex{T}
eltype_input(::TransformC2C, ::Type{T}) where {T <: FFTReal} = Complex{T}

plan(::FFT, A::AbstractArray, args...; kwargs...) = FFTW.plan_fft(A, args...; kwargs...)
plan(::FFT!, A::AbstractArray, args...; kwargs...) = FFTW.plan_fft!(A, args...; kwargs...)
plan(::BFFT, A::AbstractArray, args...; kwargs...) = FFTW.plan_bfft(A, args...; kwargs...)
plan(::BFFT!, A::AbstractArray, args...; kwargs...) = FFTW.plan_bfft!(A, args...; kwargs...)

if AMDGPU.functional(:rocfft)
    plan(::FFT, A::AnyROCArray, args...; kwargs...) = rocFFT.plan_fft(A, args...; kwargs...)
    plan(::FFT!, A::AnyROCArray, args...; kwargs...) = rocFFT.plan_fft!(A, args...; kwargs...)
    plan(::BFFT, A::AnyROCArray, args...; kwargs...) = rocFFT.plan_bfft(A, args...; kwargs...)
    plan(::BFFT!, A::AnyROCArray, args...; kwargs...) = rocFFT.plan_bfft!(A, args...; kwargs...)
end

plan(::FFT, A::AnyCuArray, args...; kwargs...) = CUFFT.plan_fft(A, args...; kwargs...)
plan(::FFT!, A::AnyCuArray, args...; kwargs...) = CUFFT.plan_fft!(A, args...; kwargs...)
plan(::BFFT, A::AnyCuArray, args...; kwargs...) = CUFFT.plan_bfft(A, args...; kwargs...)
plan(::BFFT!, A::AnyCuArray, args...; kwargs...) = CUFFT.plan_bfft!(A, args...; kwargs...)

binv(::FFT, d) = BFFT()
binv(::FFT!, d) = BFFT!()
binv(::BFFT, d) = FFT()
binv(::BFFT!, d) = FFT!()

is_inplace(::Union{FFT, BFFT}) = false
is_inplace(::Union{FFT!, BFFT!}) = true

_intprod(x::Int, y::Int...) = x * _intprod(y...)
_intprod() = one(Int)
_prod_dims(s::Dims, dims) = _intprod((s[i] for i in dims)...)
_prod_dims(A::AbstractArray, dims) = _prod_dims(size(A), dims)

scale_factor(::TransformC2C, A, dims) = _prod_dims(A, dims)
