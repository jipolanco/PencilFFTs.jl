## Permutation operations ##

# TODO
# - use TupleTools for some tuple operations? (sortperm, permute, ...)

# Permute tuple values.
permute_indices(t::NTuple, ::Nothing) = t
permute_indices(t::NTuple{N}, perm::Permutation{N}) where N = map(p -> t[p], perm)
permute_indices(t::NTuple, p::Pencil) = permute_indices(t, p.perm)

# Get "relative" permutation needed to get from `x` to `y`, i.e., such
# that `permute_indices(x, perm) == y`.
# It is assumed that both tuples have the same elements, possibly in different
# order.
function relative_permutation(x::Permutation{N}, y::Permutation{N}) where {N}
    # This is surely not the most efficient way, but it's still fast enough for
    # small tuples.
    perm = map(y) do v
        findfirst(u -> u == v, x) :: Int
    end
    # @assert permute_indices(x, perm) === y
    perm
end

relative_permutation(::Nothing, y::Permutation) = y
relative_permutation(::Nothing, ::Nothing) = nothing

# In this case, the result is the inverse permutation of `x`, such that
# `permute_indices(x, perm) == (1, 2, 3, ...)`.
relative_permutation(x::Permutation{N}, ::Nothing) where N =
    relative_permutation(x, identity_permutation(Val(N)))

relative_permutation(p::Pencil, q::Pencil) =
    relative_permutation(p.perm, q.perm)

# Construct the identity permutation: (1, 2, 3, ...)
identity_permutation(::Val{N}) where N = ntuple(identity, N)

is_identity_permutation(::Nothing) = true
is_identity_permutation(perm::Permutation{N}) where N =
    perm === identity_permutation(Val(N))
