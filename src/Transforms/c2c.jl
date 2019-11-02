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
              ::Type{Complex{T}}) where T <: FFTReal = Complex{T}

inv(::FFT) = BFFT()
inv(::IFFT) = FFT()
inv(::BFFT) = FFT()
