"""
    Transforms

Defines different one-dimensional FFT-based transforms.

The transforms are all subtypes of the [`AbstractTransform`](@ref) type.

When possible, the names of the transforms are kept consistent with the
functions exported by
[`AbstractFFTs.jl`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/)
and [`FFTW.jl`](https://juliamath.github.io/FFTW.jl/stable/fft.html).
"""
module Transforms

using FFTW
import LinearAlgebra: I

import Base: inv, show
export binv, scale_factor, normalised
export eltype_input, eltype_output, length_output, plan, expand_dims

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

Returns the (normalised) inverse of the given transform.

Note that this function is not defined for unnormalised backward transforms such
as [`BFFT`](@ref) or [`BRFFT`](@ref). For those, use [`binv`](@ref) instead.

See also [`binv`](@ref).

# Example

```jldoctest
julia> inv(Transforms.FFT())
IFFT()

julia> inv(Transforms.RFFT())
IRFFT()

julia> inv(Transforms.BRFFT())
ERROR: MethodError: no method matching inv(::PencilFFTs.Transforms.BRFFT)
```
"""
function inv end

"""
    binv(transform::AbstractTransform)

Returns the backwards transform associated to the given transform.

As opposed to [`inv`](@ref), the backwards transform returned by this function
is not normalised. The normalisation factor for a given array can be obtained
by calling [`scale_factor`](@ref).

See also [`scale_factor`](@ref), [`inv`](@ref).

# Example

```jldoctest
julia> binv(Transforms.FFT())
BFFT()

julia> binv(Transforms.BRFFT())
RFFT()

julia> binv(Transforms.IFFT())
FFT()
```
"""
function binv end

# By default, binv == inv.
binv(t::AbstractTransform) = inv(t)

"""
    scale_factor(transform::AbstractTransform, A, [dims])

Get factor required to normalise the given array after a transformation along
dimensions `dims` (all dimensions by default).

The array `A` must have the dimensions of the `transform` output.

**Important**: the dimensions `dims` must be the same that were passed to
[`plan`](@ref).

# Examples

```jldoctest
julia> C = zeros(ComplexF32, 3, 4, 5);

julia> scale_factor(Transforms.FFT(), C)
1

julia> scale_factor(Transforms.IFFT(), C)
1

julia> scale_factor(Transforms.BFFT(), C)
60

julia> scale_factor(Transforms.BFFT(), C, 2:3)
20

julia> R = zeros(Float64, 3, 4, 5);

julia> scale_factor(Transforms.BRFFT(), R, 2)
6

julia> scale_factor(Transforms.BRFFT(), R, 2:3)
30

```
"""
function scale_factor end

scale_factor(t::AbstractTransform, A) = scale_factor(t, A, 1:ndims(A))

# By default, the scale factor is 1.
scale_factor(::AbstractTransform, A, dims) = 1

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
    expand_dims(transform::AbstractTransform, Val(N))

Expand a single multidimensional transform into one transform per dimension.

# Example

```jldoctest
# Expand a real-to-complex transform in 3 dimensions.
julia> expand_dims(Transforms.RFFT(), Val(3))
(RFFT(), FFT(), FFT())

julia> expand_dims(Transforms.BRFFT(), Val(3))
(BRFFT(), BFFT(), BFFT())

julia> expand_dims(Transforms.IFFT(), Val(3))
(IFFT(), IFFT(), IFFT())

julia> expand_dims(Transforms.NoTransform(), Val(2))
(NoTransform(), NoTransform())
```
"""
function expand_dims end

expand_dims(::F, ::Val) where {F <: AbstractTransform} =
    throw(ArgumentError("I don't know how to expand transform $F"))

show(io::IO, ::F) where F <: AbstractTransform =
    # PencilFFTs.Transforms.Name -> Name()
    print(io, last(rsplit(string(F), '.', limit=2)), "()")

"""
    Normalised{B}

Trait determining whether a transform is normalised or not.

The parameter `B` is a `Bool`.

See also [`normalised`](@ref).
"""
struct Normalised{B} end

"""
    normalised(transform::Transform)

Returns [`Normalised`](@ref) trait of the given transform.
"""
function normalised end

# By default transforms are normalised.
normalised(::AbstractTransform) = Normalised{true}()

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
expand_dims(::NoTransform, ::Val{N}) where N =
    N == 0 ? () : (NoTransform(), expand_dims(NoTransform(), Val(N - 1))...)

include("c2c.jl")
include("r2c.jl")

end
