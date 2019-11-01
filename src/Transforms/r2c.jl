## Real-to-complex and complex-to-real transforms.

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
