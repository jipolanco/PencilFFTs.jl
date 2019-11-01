## Complex-to-complex transforms.

"""
    FFT()

Complex-to-complex FFT.

See also
[`AbstractFFTs.fft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.fft).
"""
struct FFT <: AbstractTransform end
inv(::FFT) = BFFT()

"""
    IFFT()

Normalised inverse complex-to-complex FFT.

See also
[`AbstractFFTs.ifft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.ifft).
"""
struct IFFT <: AbstractTransform end
inv(::IFFT) = FFT()

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
inv(::BFFT) = FFT()
