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

using FFTW
import LinearAlgebra: I

import Base: inv
export eltype_input, eltype_output, length_output, plan

const FFTReal = FFTW.fftwReal  # = Union{Float32, Float64}

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
    plan(transform::AbstractTransform, A, [dims];
         flags=FFTW.ESTIMATE, timelimit=Inf)

Create plan to transform array `A` along dimensions `dims`.

If `dims` is not specified, all dimensions of `A` are transformed.

This function wraps the `AbstractFFTs.jl` and `FFTW.jl` plan creation functions.
For more details on the function arguments, see
[`AbstractFFTs.plan_fft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.plan_fft).
"""
function plan end

plan(t::AbstractTransform, A; kwargs...) = plan(t, A, 1:ndims(A); kwargs...)

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
    eltype_input(transform::AbstractTransform, real_type<:AbstractFloat)

Determine input data type for a given transform given the floating point
precision of the input data.

For some transforms such as `NoTransform`, the input type cannot be identified
only from `real_type`. In this case, `Nothing` is returned.

# Example

```jldoctest
julia> eltype_input(Transforms.FFT(), Float32)
Complex{Float32}

julia> eltype_input(Transforms.RFFT(), Float64)
Float64

julia> eltype_input(Transforms.NoTransform(), Float64)
Nothing

```
"""
function eltype_input end

"""
    eltype_output(transform::AbstractTransform, eltype_input)

Returns the output data type for a given transform given the input type.

Throws `ArgumentError` if the input data type is incompatible with the transform
type.

# Example

```jldoctest
julia> eltype_output(Transforms.NoTransform(), Float32)
Float32

julia> eltype_output(Transforms.RFFT(), Float64)
Complex{Float64}

julia> eltype_output(Transforms.BRFFT(), ComplexF32)
Float32

julia> eltype_output(Transforms.FFT(), Float64)
ERROR: ArgumentError: invalid input data type for PencilFFTs.Transforms.FFT: Float64
```
"""
function eltype_output end

eltype_output(::F, ::Type{T}) where {F <: AbstractTransform, T} =
    throw(ArgumentError("invalid input data type for $F: $T"))

"""
    NoTransform()

Identity transform.

Specifies that no transformation should be applied.
"""
struct NoTransform <: AbstractTransform end
inv(::NoTransform) = NoTransform()
length_output(::NoTransform, length_in::Integer) = length_in
eltype_output(::NoTransform, ::Type{T}) where T = T
eltype_input(::NoTransform, ::Type) = Nothing
plan(::NoTransform, A, dims; kwargs...) = I  # identity matrix (UniformScaling)

include("c2c.jl")
include("r2c.jl")

end
