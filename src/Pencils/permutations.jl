## Permutation operations ##

# TODO define AbstractArray wrapper to tuples, to call sortperm

# Permute tuple values.
permute_indices(t::NTuple, ::Nothing) = t
permute_indices(t::NTuple{N}, perm::Permutation{N}) where N = map(p -> t[p], perm)
permute_indices(t::NTuple, p::Pencil) = permute_indices(t, p.perm)

# Get "relative" permutation needed to get from `x` to `y`, i.e., such
# that `permute_indices(x, perm) == y`.
# It is **assumed** that both tuples have the same elements, possibly in
# different order.
function relative_permutation(x::Permutation{N}, y::Permutation{N}) where {N}
    # TODO There must be a better algorithm for this...
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
relative_permutation(x::Permutation{N}, ::Nothing) where N =
    relative_permutation(x, ntuple(n -> n, N))  # TODO better way to do this?

relative_permutation(p::Pencil, q::Pencil) =
    relative_permutation(p.perm, q.perm)
