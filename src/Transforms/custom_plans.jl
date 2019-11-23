"""
    IdentityPlan

Type of plan associated to [`NoTransform`](@ref).
"""
struct IdentityPlan end

LinearAlgebra.mul!(y, ::IdentityPlan, x) = (y === x) ? y : copy!(y, x)
LinearAlgebra.ldiv!(y, ::IdentityPlan, x) = mul!(y, IdentityPlan(), x)
Base.:*(::IdentityPlan, x) = copy(x)
Base.:\(::IdentityPlan, x) = copy(x)
