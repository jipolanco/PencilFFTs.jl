"""
Defines different one-dimensional FFT-based transforms.

The transforms are all subtypes of an [`AbstractTransform`](@ref) type.

When possible, the names of the transforms are kept consistent with the
functions exported by
[`AbstractFFTs.jl`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/)
and [`FFTW.jl`](https://juliamath.github.io/FFTW.jl/stable/fft.html).
"""
module Transforms

using FFTW

# Operations defined for custom plans (currently IdentityPlan).
import LinearAlgebra: mul!, ldiv!
import Base: *, \

import Base: show
export binv, scale_factor
export eltype_input, eltype_output, length_output, plan, expand_dims

const FFTReal = FFTW.fftwReal  # = Union{Float32, Float64}
const RealArray{T} = AbstractArray{T} where T <: FFTReal
const ComplexArray{T} = AbstractArray{T} where T <: Complex

# TODO
# - add FFTW.jl specific transforms, including r2r
#   (see https://juliamath.github.io/FFTW.jl/stable/fft.html)

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
    binv(transform::AbstractTransform)

Returns the backwards transform associated to the given transform.

The backwards transform returned by this function is not normalised. The
normalisation factor for a given array can be obtained by calling
[`scale_factor`](@ref).

# Example

```jldoctest
julia> binv(Transforms.FFT())
BFFT()

julia> binv(Transforms.BRFFT())
RFFT()
```
"""
function binv end

"""
    scale_factor(transform::AbstractTransform, A, [dims])

Get factor required to normalise the given array after a transformation along
dimensions `dims` (all dimensions by default).

The array `A` must have the dimensions of the `transform` output.

**Important**: the dimensions `dims` must be the same that were passed to
[`plan`](@ref).

# Examples

```jldoctest scale_factor
julia> C = zeros(ComplexF32, 3, 4, 5);

julia> scale_factor(Transforms.FFT(), C)
60

julia> scale_factor(Transforms.BFFT(), C)
60

julia> scale_factor(Transforms.BFFT(), C, 2:3)
20

julia> R = zeros(Float64, 3, 4, 5);

julia> scale_factor(Transforms.BRFFT(), R, 2)
4

julia> scale_factor(Transforms.BRFFT(), R, 2:3)
20
```

This will fail because the output of `RFFT` is complex, and `R` is a real array:
```jldoctest scale_factor
julia> scale_factor(Transforms.RFFT(), R, 2:3)
ERROR: MethodError: no method matching scale_factor(::PencilFFTs.Transforms.RFFT, ::Array{Float64,3}, ::UnitRange{Int64})
```
"""
function scale_factor end

scale_factor(t::AbstractTransform, A) = scale_factor(t, A, 1:ndims(A))

"""
    length_output(transform::AbstractTransform, length_in::Integer)

Returns the length of the transform output, given the length of its input.

The input and output lengths are specified in terms of the respective input
and output datatypes.
For instance, for real-to-complex transforms, these are respectively the
length of input *real* data and of output *complex* data.

Also note that for backward real-to-complex transforms ([`BRFFT`](@ref)), it is
assumed that the real data length is even. See also
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
    NoTransform()

Identity transform.

Specifies that no transformation should be applied.
"""
struct NoTransform <: AbstractTransform end
binv(::NoTransform) = NoTransform()
length_output(::NoTransform, length_in::Integer) = length_in
eltype_output(::NoTransform, ::Type{T}) where T = T
eltype_input(::NoTransform, ::Type) = Nothing
plan(::NoTransform, A, dims; kwargs...) = IdentityPlan()
expand_dims(::NoTransform, ::Val{N}) where N =
    N == 0 ? () : (NoTransform(), expand_dims(NoTransform(), Val(N - 1))...)
scale_factor(::NoTransform, A, dims) = 1

include("c2c.jl")
include("r2c.jl")
include("custom_plans.jl")

end
