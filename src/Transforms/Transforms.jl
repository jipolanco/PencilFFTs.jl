"""
    Transforms

Defines different one-dimensional FFT-based transforms.

The transforms are all subtypes of an `AbstractTransform` type.

When possible, the names of the transforms are kept consistent with the
functions exported by `AbstractFFTs.jl` and `FFTW.jl`.
"""
module Transforms

# TODO
# - add FFTW.jl specific transforms, including r2r
#   (see https://juliamath.github.io/FFTW.jl/stable/fft.html)
# - Chebyshev as an alias for r2r? (with kind = REDFT00, I think...)

"""
    AbstractTransform

Specifies a one-dimensional FFT-based transform.
"""
abstract type AbstractTransform end

"""
    NoTransform()

Specifies that no transformation should be applied.
"""
struct NoTransform <: AbstractTransform end

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

end
