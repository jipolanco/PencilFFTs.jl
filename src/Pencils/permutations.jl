## Permutation operations ##

const Permutation{N} = NTuple{N,Int} where N

# Permute tuple values.
@inline permute_indices(t::Tuple, ::Nothing) = t
@inline permute_indices(t::Tuple, p::Pencil) = permute_indices(t, p.perm)
@inline function permute_indices(t::Tuple{Vararg{Any,N}},
                                 ::Val{perm}) where {N, perm}
    perm :: Permutation{N}
    @inbounds ntuple(i -> t[perm[i]], Val(N))
end
@inline permute_indices(::Val{t}, p::Val) where {t} = Val(permute_indices(t, p))

@inline permute_indices(I::CartesianIndex, perm) =
    CartesianIndex(permute_indices(Tuple(I), perm))

# Get "relative" permutation needed to get from `x` to `y`, i.e., such
# that `permute_indices(x, perm) == y`.
# It is assumed that both tuples have the same elements, possibly in different
# order.
function relative_permutation(x::Val{p}, y::Val{q}) where {p, q}
    N = length(p)
    p :: Permutation{N}
    q :: Permutation{N}
    if @generated
        perm = map(v -> findfirst(==(v), p)::Int, q)
        @assert permute_indices(p, Val(perm)) === q
        :( Val($perm) )
    else
        perm = map(v -> findfirst(==(v), p)::Int, q)
        @assert permute_indices(p, Val(perm)) === q
        Val(perm)
    end
end

relative_permutation(::Nothing, y::Val) = y
relative_permutation(::Nothing, ::Nothing) = nothing

# In this case, the result is the inverse permutation of `x`, such that
# `permute_indices(x, perm) == (1, 2, 3, ...)`.
# (Same as `invperm`, which is type unstable for tuples.)
relative_permutation(x::Val{p}, ::Nothing) where {p} =
    relative_permutation(x, identity_permutation(Val(length(p))))

inverse_permutation(x::Union{Nothing, Val}) = relative_permutation(x, nothing)

relative_permutation(p::Pencil, q::Pencil) =
    relative_permutation(p.perm, q.perm)

# Construct the identity permutation: (1, 2, 3, ...)
identity_permutation(::Val{N}) where N = Val(ntuple(identity, N))

is_identity_permutation(::Nothing) = true

function is_identity_permutation(::Val{P}) where P
    N = length(P)
    P :: Permutation{N}
    P === identity_permutation(Val(N))
end

is_valid_permutation(::Nothing) = true
is_valid_permutation(::Val{P}) where {P} = isa(P, Permutation) && isperm(P)
is_valid_permutation(::Any) = false

same_permutation(::Val{p}, ::Val{p}) where {p} = true
same_permutation(::Val{p}, ::Val{q}) where {p, q} = (@assert p !== q; false)
same_permutation(::Val{p}, ::Nothing) where {p} =
    p === identity_permutation(Val(length(p)))
same_permutation(::Nothing, p::Val) = same_permutation(p, nothing)
same_permutation(::Nothing, ::Nothing) = true

# Append `M` non-permuted dimensions to the given permutation.
# Example: append_to_permutation(Val((2, 3, 1)), Val(2)) = Val((2, 3, 1, 4, 5)).
function append_to_permutation(::Val{p}, ::Val{M}) where {p, M}
    N = length(p)
    p :: Permutation{N}
    Val((p..., ntuple(i -> N + i, Val(M))...))
end

# This is useful for base functions that don't accept permutations as value
# types (like `permutedims!`).
extract(::Val{p}) where {p} = p
