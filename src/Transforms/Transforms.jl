"""
    Transforms

Defines different one-dimensional FFT-based transforms.

The transforms are all subtypes of the [`AbstractTransform`](@ref) type.

When possible, the names of the transforms are kept consistent with the
functions exported by
[`AbstractFFTs.jl`](https://juliamath.github.io/AbstractFFTs.jl/stable/api)
and [`FFTW.jl`](https://juliamath.github.io/FFTW.jl/stable/fft.html).
"""
module Transforms

import Base: inv

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
    inv(transform::AbstractTransform)

Returns the (unnormalised) inverse of the given transform.

Note that there is no one-to-one correspondence between direct and inverse
transforms. For instance, inverse of [`FFT`](@ref) is [`BFFT`](@ref), while
[`FFT`](@ref) is the inverse of both [`BFFT`](@ref) and [`IFFT`](@ref).
"""
function inv end

"""
    NoTransform()

Identity transform.

Specifies that no transformation should be applied.
"""
struct NoTransform <: AbstractTransform end
inv(::NoTransform) = NoTransform()

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

"""
    RFFT()

Real-to-complex FFT.

See also
[`AbstractFFTs.rfft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.rfft).
"""
struct RFFT <: AbstractTransform end
inv(::RFFT) = BRFFT()

"""
    IRFFT()

Normalised inverse of [`RFFT`](@ref).

See also
[`AbstractFFTs.irfft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.irfft).
"""
struct IRFFT <: AbstractTransform end
inv(::IRFFT) = RFFT()

"""
    BRFFT()

Unnormalised inverse of [`RFFT`](@ref).

To obtain the inverse transform, divide the output by the length of the
transformed dimension (of the real output array).

See also
[`AbstractFFTs.brfft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.brfft).
"""
struct BRFFT <: AbstractTransform end
inv(::BRFFT) = RFFT()

end
