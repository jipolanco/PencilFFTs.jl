"""
Defines different one-dimensional FFT-based transforms.

The transforms are all subtypes of an [`AbstractTransform`](@ref) type.

When possible, the names of the transforms are kept consistent with the
functions exported by
[`AbstractFFTs.jl`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/)
and [`FFTW.jl`](https://juliamath.github.io/FFTW.jl/stable/fft/).
"""
module Transforms

using FFTW

# Operations defined for custom plans (currently IdentityPlan).
using LinearAlgebra

export binv, scale_factor, is_inplace
export eltype_input, eltype_output, length_output, plan, expand_dims

const FFTReal = FFTW.fftwReal  # = Union{Float32, Float64}
const RealArray{T} = AbstractArray{T} where T <: FFTReal
const ComplexArray{T} = AbstractArray{T} where T <: Complex

"""
    AbstractTransform

Specifies a one-dimensional FFT-based transform.
"""
abstract type AbstractTransform end

"""
    AbstractCustomPlan

Abstract type defining a custom plan, to be used as an alternative to FFTW
plans (`FFTW.FFTWPlan`).

The only custom plan defined in this module is [`IdentityPlan`](@ref).
The user can define other custom plans that are also subtypes of
`AbstractCustomPlan`.

Note that [`plan`](@ref) returns a subtype of either `FFTW.FFTWPlan` or
`AbstractCustomPlan`.
"""
abstract type AbstractCustomPlan end

"""
    Plan = Union{FFTW.FFTWPlan, AbstractCustomPlan}

Union type representing any plan returned by [`plan`](@ref).

See also [`AbstractCustomPlan`](@ref).
"""
const Plan = Union{FFTW.FFTWPlan, AbstractCustomPlan}

"""
    plan(transform::AbstractTransform, A, [dims];
         flags=FFTW.ESTIMATE, timelimit=Inf)

Create plan to transform array `A` along dimensions `dims`.

If `dims` is not specified, all dimensions of `A` are transformed.

For FFT plans, this function wraps the `AbstractFFTs.jl` and `FFTW.jl` plan
creation functions.
For more details on the function arguments, see
[`AbstractFFTs.plan_fft`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.plan_fft).
"""
function plan end

function plan(t::AbstractTransform, A; kwargs...)
    # Instead of passing dims = 1:N, we pass a tuple (1, 2, ..., N) to make sure
    # that the length of dims is known at compile time. This is important for
    # guessing the return type of r2r plans, which in principle are type
    # unstable (see comments in r2r.jl).
    N = ndims(A)
    dims = ntuple(identity, Val(N))  # (1, 2, ..., N)
    plan(t, A, dims; kwargs...)
end

"""
    binv(transform::AbstractTransform, d::Integer)

Returns the backwards transform associated to the given transform.

The second argument must be the length of the first transformed dimension in
the forward transform.
It is used in particular when `transform = RFFT()`, to determine the length of
the inverse (complex-to-real) transform.
See the [`AbstractFFTs.irfft` docs](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.irfft)
for details.

The backwards transform returned by this function is not normalised. The
normalisation factor for a given array can be obtained by calling
[`scale_factor`](@ref).

# Example

```jldoctest
julia> binv(Transforms.FFT(), 42)
BFFT

julia> binv(Transforms.BRFFT(), 42)
RFFT
```
"""
function binv end

"""
    is_inplace(transform::AbstractTransform)         -> Bool
    is_inplace(transforms::Vararg{AbtractTransform}) -> Union{Bool, Nothing}

Check whether a transform or a list of transforms is performed in-place.

If the list of transforms has a combination of in-place and out-of-place
transforms, `nothing` is returned.

# Example

```jldoctest; setup = :(import FFTW)
julia> is_inplace(Transforms.RFFT())
false

julia> is_inplace(Transforms.NoTransform!())
true

julia> is_inplace(Transforms.FFT!(), Transforms.R2R!(FFTW.REDFT01))
true

julia> is_inplace(Transforms.FFT(), Transforms.R2R(FFTW.REDFT01))
false

julia> is_inplace(Transforms.FFT(), Transforms.R2R!(FFTW.REDFT01)) === nothing
true

```
"""
function is_inplace end

@inline function is_inplace(tr::AbstractTransform, tr2::AbstractTransform,
                            next::Vararg{AbstractTransform})
    b = is_inplace(tr2, next...)
    b === nothing && return nothing
    a = is_inplace(tr)
    a === b ? a : nothing
end

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
ERROR: MethodError: no method matching scale_factor(::PencilFFTs.Transforms.RFFT, ::Array{Float64, 3}, ::UnitRange{Int64})
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
"""
function length_output end

"""
    eltype_input(transform::AbstractTransform, real_type<:AbstractFloat)

Determine input data type for a given transform given the floating point
precision of the input data.

Some transforms, such as [`R2R`](@ref) and [`NoTransform`](@ref), can take both
real and complex data. For those kinds of transforms, `nothing` is returned.

# Example

```jldoctest; setup = :(import FFTW)
julia> eltype_input(Transforms.FFT(), Float32)
ComplexF32 (alias for Complex{Float32})

julia> eltype_input(Transforms.RFFT(), Float64)
Float64

julia> eltype_input(Transforms.R2R(FFTW.REDFT01), Float64)
nothing

julia> eltype_input(Transforms.NoTransform(), Float64)
nothing

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
ComplexF64 (alias for Complex{Float64})

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
(RFFT, FFT, FFT)

julia> expand_dims(Transforms.BRFFT(), Val(3))
(BRFFT{even}, BFFT, BFFT)

julia> expand_dims(Transforms.NoTransform(), Val(2))
(NoTransform, NoTransform)
```
"""
function expand_dims end

expand_dims(::F, ::Val) where {F <: AbstractTransform} =
    throw(ArgumentError("I don't know how to expand transform $F"))

function Base.show(io::IO, tr::F) where {F <: AbstractTransform}
    print(io, nameof(F))
    _show_extra_info(io, tr)
end

_show_extra_info(::IO, ::AbstractTransform) = nothing

include("c2c.jl")
include("r2c.jl")
include("r2r.jl")
include("no_transform.jl")

end
