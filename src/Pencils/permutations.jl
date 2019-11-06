## Permutation operations ##

# Permute tuple values.
permute_indices(t::NTuple, ::Nothing) = t
permute_indices(t::NTuple{N}, perm::Permutation{N}) where N = map(p -> t[p], perm)
permute_indices(t::NTuple, p::Pencil) = permute_indices(t, p.perm)

permute_indices(I::CartesianIndex, perm) =
    CartesianIndex(permute_indices(Tuple(I), perm))

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
    @assert permute_indices(x, perm) === y
    perm
end

relative_permutation(::Nothing, y::Permutation) = y
relative_permutation(::Nothing, ::Nothing) = nothing

# In this case, the result is the inverse permutation of `x`, such that
# `permute_indices(x, perm) == (1, 2, 3, ...)`.
# TODO compare to using `invperm`
relative_permutation(x::Permutation{N}, ::Nothing) where N =
    relative_permutation(x, identity_permutation(Val(N)))

relative_permutation(p::Pencil, q::Pencil) =
    relative_permutation(p.perm, q.perm)

# Construct the identity permutation: (1, 2, 3, ...)
identity_permutation(::Val{N}) where N = ntuple(identity, N)

is_identity_permutation(::Nothing) = true
is_identity_permutation(perm::Permutation{N}) where N =
    perm === identity_permutation(Val(N))

is_valid_permuation(::Nothing) = true
is_valid_permuation(perm::Permutation) = isperm(perm)

same_permutation(a::P, b::P) where P = a === b
same_permutation(a::Permutation{N}, ::Nothing) where N =
    a === identity_permutation(Val(N))
same_permutation(::Nothing, a::Permutation) = same_permutation(a, nothing)

# Prepend `M` non-permuted dimensions to the given permutation.
# Example: prepend_to_permutation(Val(2), (2, 3, 1)) = (1, 2, 4, 5, 3).
prepend_to_permutation(::Val{M}, perm::Permutation) where M =
    (ntuple(identity, M)..., (M .+ perm)...)
