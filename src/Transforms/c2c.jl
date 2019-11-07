## Complex-to-complex transforms.

"""
    FFT()

Complex-to-complex FFT.

See also
[`AbstractFFTs.fft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.fft).
"""
struct FFT <: AbstractTransform end

"""
    IFFT()

Normalised inverse complex-to-complex FFT.

See also
[`AbstractFFTs.ifft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.ifft).
"""
struct IFFT <: AbstractTransform end

"""
    BFFT()

Unnormalised inverse (backward) complex-to-complex FFT.

Like `AbstractFFTs.bfft`, this transform is not normalised.
To obtain the inverse transform, divide the output by the length of the
transformed dimension.

See also
[`AbstractFFTs.bfft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.bfft).
"""
struct BFFT <: AbstractTransform end

const TransformC2C = Union{FFT, IFFT, BFFT}

length_output(::TransformC2C, length_in::Integer) = length_in
eltype_output(::TransformC2C,
              ::Type{Complex{T}}) where {T <: FFTReal} = Complex{T}
eltype_input(::TransformC2C, ::Type{T}) where {T <: FFTReal} = Complex{T}

plan(::FFT, args...; kwargs...) = FFTW.plan_fft(args...; kwargs...)
plan(::IFFT, args...; kwargs...) = FFTW.plan_ifft(args...; kwargs...)
plan(::BFFT, args...; kwargs...) = FFTW.plan_bfft(args...; kwargs...)

inv(::FFT) = BFFT()
inv(::IFFT) = FFT()
inv(::BFFT) = FFT()

split_dims(::F, ::Val{N}) where {F <: TransformC2C, N} =
    N === 0 ? () : (F(), split_dims(F(), Val(N - 1))...)
