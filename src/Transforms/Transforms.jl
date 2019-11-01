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
    length_output(transform::AbstractTransform, length_in::Integer)

Returns the length of the transform output, given the length of its input.

The input and output lengths are specified in terms of the respective input
and output datatypes.
For instance, for real-to-complex transforms, these are respectively the
length of input *real* data and of output *complex* data.

Also note that for inverse real-to-complex transforms ([`IRFFT`](@ref) and
[`BRFFT`](@ref)), it is assumed that the real data length is even. See also
the [`AbstractFFTs.irfft`
docs](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.irfft).
"""
function length_output end

"""
    NoTransform()

Identity transform.

Specifies that no transformation should be applied.
"""
struct NoTransform <: AbstractTransform end
inv(::NoTransform) = NoTransform()
length_output(::NoTransform, length_in::Integer) = length_in

include("c2c.jl")
include("r2c.jl")

end
