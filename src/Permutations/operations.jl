## Permutation operations ##

# Extract tuple representation of Permutation.
# The result can be passed to functions like permutedims.
Base.Tuple(::Permutation{p}) where {p} = p
Base.Tuple(::NoPermutation) = error("cannot convert NoPermutation to tuple")

Base.length(::Permutation{p}) where {p} = length(p)

is_valid_permutation(::NoPermutation) = true
is_valid_permutation(::Permutation{P}) where {P} = isperm(P)

"""
    check_permutation(perm)

Check the validity of a `Permutation`.

Throws `ArgumentError` if the permutation is invalid.

# Examples

```jldoctest
julia> check_permutation(Permutation(3, 2, 1))  # no error

julia> check_permutation(NoPermutation())       # no error

julia> check_permutation(Permutation(3, 3, 1))
ERROR: ArgumentError: invalid permutation of dimensions: Permutation(3, 3, 1)
```
"""
function check_permutation(perm)
    is_valid_permutation(perm) && return
    throw(ArgumentError("invalid permutation of dimensions: $perm"))
end

"""
    permute_indices(indices, perm::Permutation)

Permute indices according to a compile-time permutation.

`indices` may be a `Tuple` of indices, a `CartesianIndex`, or a `Permutation` to
be reordered according to `perm`.

# Examples

```jldoctest
julia> perm = Permutation(2, 3, 1);

julia> permute_indices((36, 42, 14), perm)
(42, 14, 36)

julia> permute_indices(CartesianIndex(36, 42, 14), perm)
CartesianIndex(42, 14, 36)

julia> permute_indices(Permutation(3, 1, 2), perm)
Permutation(1, 2, 3)
```
"""
@inline permute_indices(t::Tuple, ::NoPermutation) = t
@inline function permute_indices(t::Tuple{Vararg{Any,N}},
                                 ::Permutation{perm,N}) where {N, perm}
    @inbounds ntuple(i -> t[perm[i]], Val(N))
end
@inline permute_indices(::Permutation{t}, p::Permutation) where {t} =
    Permutation(permute_indices(t, p)...)

@inline permute_indices(I::CartesianIndex, perm) =
    CartesianIndex(permute_indices(Tuple(I), perm))

"""
    relative_permutation(x::Permutation, y::Permutation)

Get relative permutation needed to get from `x` to `y`.
That is, the permutation `perm` such that `permute_indices(x, perm) == y`.

The computation is performed at compile time using generated functions.

# Examples

```jldoctest
julia> x = Permutation(3, 1, 2);

julia> y = Permutation(2, 1, 3);

julia> perm = relative_permutation(x, y)
Permutation(3, 2, 1)

julia> permute_indices(x, perm) == y
true
```
"""
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

"""
    inverse_permutation(p::Permutation)

Returns the inverse permutation of `p`.

Functionally equivalent to Julia's `invperm`, with the advantage that the result
is a compile time constant.

See also [`relative_permutation`](@ref).

# Examples

```jldoctest
julia> p = Permutation(2, 3, 1);

julia> q = inverse_permutation(p)
Permutation(3, 1, 2)

julia> t_orig = (36, 42, 14);

julia> t_perm = permute_indices(t_orig, p)
(42, 14, 36)

julia> permute_indices(t_perm, q) === t_orig
true

```
"""
inverse_permutation(x::Permutation) = relative_permutation(x, NoPermutation())

# Construct the identity permutation: (1, 2, 3, ...)
identity_permutation(::Val{N}) where {N} = Permutation(ntuple(identity, Val(N)))

"""
    is_identity_permutation(p::Permutation)

Returns `true` if `p` is an identity permutation, i.e. if it is equivalent to
`(1, 2, 3, ...)`.

```jldoctest
julia> is_identity_permutation(Permutation(1, 2, 3))
true

julia> is_identity_permutation(Permutation(1, 3, 2))
false

julia> is_identity_permutation(NoPermutation())
true
```
"""
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
"""
    append_to_permutation(p::Permutation, ::Val{M})

Append `M` non-permuted dimensions to the given permutation.

# Examples

```jldoctest
julia> append_to_permutation(Permutation(2, 3, 1), Val(2))
Permutation(2, 3, 1, 4, 5)

julia> append_to_permutation(NoPermutation(), Val(2))
NoPermutation()
```
"""
function append_to_permutation(::Permutation{p}, ::Val{M}) where {p, M}
    N = length(p)
    Permutation(p..., ntuple(i -> N + i, Val(M))...)
end

append_to_permutation(np::NoPermutation, ::Val) = np
