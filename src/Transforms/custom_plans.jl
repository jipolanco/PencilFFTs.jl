"""
    IdentityPlan

Type of plan associated to [`NoTransform`](@ref).
"""
struct IdentityPlan end

mul!(y, ::IdentityPlan, x) = (y === x) ? y : copy!(y, x)
ldiv!(y, ::IdentityPlan, x) = mul!(y, IdentityPlan(), x)
*(::IdentityPlan, x) = copy(x)
\(::IdentityPlan, x) = copy(x)
