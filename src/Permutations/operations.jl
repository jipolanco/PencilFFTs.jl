## Permutation operations ##

# Extract tuple representation of Permutation.
# The result can be passed to functions like permutedims.
Base.Tuple(::Permutation{p}) where {p} = p
Base.Tuple(::NoPermutation) = error("cannot convert NoPermutation to tuple")

Base.length(::Permutation{p}) where {p} = length(p)

is_valid_permutation(::NoPermutation) = true
is_valid_permutation(::Permutation{P}) where {P} = isperm(P)

function check_permutation(perm)
    is_valid_permutation(perm) && return
    throw(ArgumentError("invalid permutation of dimensions: $perm"))
end

# Permute tuple values.
@inline permute_indices(t::Tuple, ::NoPermutation) = t
@inline function permute_indices(t::Tuple{Vararg{Any,N}},
                                 ::Permutation{perm,N}) where {N, perm}
    @inbounds ntuple(i -> t[perm[i]], Val(N))
end
@inline permute_indices(::Permutation{t}, p::Permutation) where {t} =
    Permutation(permute_indices(t, p)...)

@inline permute_indices(I::CartesianIndex, perm) =
    CartesianIndex(permute_indices(Tuple(I), perm))

# Get "relative" permutation needed to get from `x` to `y`, i.e., such
# that `permute_indices(x, perm) == y`.
# It is assumed that both tuples have the same elements, possibly in different
# order.
function relative_permutation(::Permutation{p,N},
                              ::Permutation{q,N}) where {p, q, N}
    if @generated
        perm = map(v -> findfirst(==(v), p)::Int, q)
        @assert permute_indices(p, Permutation(perm)) === q
        :( Permutation($perm) )
    else
        perm = map(v -> findfirst(==(v), p)::Int, q)
        @assert permute_indices(p, Permutation(perm)) === q
        Permutation(perm)
    end
end

relative_permutation(::NoPermutation, y::Permutation) = y
relative_permutation(::NoPermutation, y::NoPermutation) = y

# In this case, the result is the inverse permutation of `x`, such that
# `permute_indices(x, perm) == (1, 2, 3, ...)`.
# (Same as `invperm`, which is type unstable for tuples.)
relative_permutation(x::Permutation{p}, ::NoPermutation) where {p} =
    relative_permutation(x, identity_permutation(Val(length(p))))

inverse_permutation(x::Permutation) = relative_permutation(x, NoPermutation())

# Construct the identity permutation: (1, 2, 3, ...)
identity_permutation(::Val{N}) where N = Permutation(ntuple(identity, N))

is_identity_permutation(::NoPermutation) = true

function is_identity_permutation(perm::Permutation)
    N = length(perm)
    perm === identity_permutation(Val(N))
end

# Comparisons: (1, 2, ..., N) is considered equal to NoPermutation, for any N.
Base.:(==)(::Permutation{p}, ::Permutation{q}) where {p, q} = p === q
Base.:(==)(::NoPermutation, ::NoPermutation) = true
Base.:(==)(p::Permutation, ::NoPermutation) = is_identity_permutation(p)
Base.:(==)(np::NoPermutation, p::Permutation) = p == np

# Append `M` non-permuted dimensions to the given permutation.
# Example: append_to_permutation(Permutation((2, 3, 1)), Val(2)) =
# Permutation((2, 3, 1, 4, 5)).
function append_to_permutation(::Permutation{p}, ::Val{M}) where {p, M}
    N = length(p)
    Permutation(p..., ntuple(i -> N + i, Val(M))...)
end

append_to_permutation(np::NoPermutation, ::Val) = np
