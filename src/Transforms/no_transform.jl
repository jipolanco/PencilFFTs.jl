"""
    NoTransform()

Identity transform.

Specifies that no transformation should be applied.
"""
struct NoTransform <: AbstractTransform end

"""
    NoTransform!()

In-place version of [`NoTransform`](@ref).
"""
struct NoTransform! <: AbstractTransform end

const AnyNoTransform = Union{NoTransform, NoTransform!}

is_inplace(::NoTransform) = false
is_inplace(::NoTransform!) = true

binv(::T, d) where {T <: AnyNoTransform} = T()
length_output(::AnyNoTransform, length_in::Integer) = length_in
eltype_output(::AnyNoTransform, ::Type{T}) where T = T
eltype_input(::AnyNoTransform, ::Type) = nothing
scale_factor(::AnyNoTransform, A, dims) = 1

plan(::NoTransform, A, dims; kwargs...) = IdentityPlan()
plan(::NoTransform!, A, dims; kwargs...) = IdentityPlan!()

"""
    IdentityPlan

Type of plan associated to [`NoTransform`](@ref).
"""
struct IdentityPlan <: AbstractCustomPlan end

LinearAlgebra.mul!(y, ::IdentityPlan, x) = (y === x) ? y : copy!(y, x)
LinearAlgebra.ldiv!(y, ::IdentityPlan, x) = mul!(y, IdentityPlan(), x)
Base.:*(::IdentityPlan, x) = copy(x)
Base.:\(::IdentityPlan, x) = copy(x)

"""
    IdentityPlan!

Type of plan associated to [`NoTransform!`](@ref).
"""
struct IdentityPlan! <: AbstractCustomPlan end

function LinearAlgebra.mul!(y, ::IdentityPlan!, x)
    if x !== y
        throw(ArgumentError("in-place IdentityPlan applied to out-of-place data"))
    end
    y
end
LinearAlgebra.ldiv!(y, ::IdentityPlan!, x) = mul!(y, IdentityPlan!(), x)
Base.:*(::IdentityPlan!, x) = x
Base.:\(::IdentityPlan!, x) = x
