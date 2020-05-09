# Custom definitions of LinearIndices and CartesianIndices to take into account
# index permutations.
#
# In particular, when array dimensions are permuted, the default
# CartesianIndices do not iterate in memory order, making them suboptimal.
# We try to workaround that by adding a custom definition of CartesianIndices.
#
# (TODO Better / cleaner way to do this??)

# We make LinearIndices(::PencilArray) return a PermutedLinearIndices, which
# takes index permutation into account.
struct PermutedLinearIndices{
        N, L <: LinearIndices, Perm,
        Offsets <: Union{Nothing, Dims{N}},
    } <: AbstractVector{Int}
    data :: L  # indices in permuted order
    perm :: Perm
    offsets :: Offsets
    function PermutedLinearIndices(ind::LinearIndices{N},
                                   perm::Perm, offsets=nothing) where {N, Perm}
        L = typeof(ind)
        Off = typeof(offsets)
        new{N, L, Perm, Off}(ind, perm, offsets)
    end
end

Base.length(L::PermutedLinearIndices) = length(L.data)
Base.size(L::PermutedLinearIndices) = (length(L), )
Base.iterate(L::PermutedLinearIndices, args...) = iterate(L.data, args...)
Base.lastindex(L::PermutedLinearIndices) = lastindex(L.data)

@inline _apply_offset(I::CartesianIndex, ::Nothing) = I
@inline _apply_offset(I::CartesianIndex{N}, off::Dims{N}) where {N} =
    CartesianIndex(Tuple(I) .- off)

@inline function Base.getindex(L::PermutedLinearIndices, i::Integer)
    @boundscheck checkbounds(L.data, i)
    @inbounds L.data[i]
end

@inline function Base.getindex(
        L::PermutedLinearIndices{N}, I::CartesianIndex{N}) where {N}
    Ioff = _apply_offset(I, L.offsets)
    J = permute_indices(Ioff, L.perm)
    @boundscheck checkbounds(L.data, J)
    @inbounds L.data[J]
end

Base.LinearIndices(A::PencilArray) =
    PermutedLinearIndices(LinearIndices(parent(A)), get_permutation(A))

function Base.LinearIndices(g::GlobalPencilArray)
    A = parent(g)
    off = g.offsets
    PermutedLinearIndices(LinearIndices(parent(A)), get_permutation(A), off)
end

# We make CartesianIndices(::PencilArray) return a PermutedCartesianIndices,
# which loops faster (in memory order) when there are index permutations.
struct PermutedCartesianIndices{
        N, C <: CartesianIndices{N}, Iperm,
        Offsets} <: AbstractArray{CartesianIndex{N}, N}
    data  :: C  # indices in permuted order
    iperm :: Iperm  # inverse permutation
    offsets :: Offsets
    function PermutedCartesianIndices(ind::CartesianIndices{N},
                                      perm::Perm, offsets=nothing) where {N, Perm}
        iperm = inverse_permutation(perm)
        C = typeof(ind)
        Iperm = typeof(iperm)
        Off = typeof(offsets)
        new{N, C, Iperm, Off}(ind, iperm, offsets)
    end
end

@inline function Base.iterate(C::PermutedCartesianIndices, args...)
    next = iterate(C.data, args...)
    next === nothing && return nothing
    I, state = next                  # `I` has permuted indices
    J = permute_indices(I, C.iperm)  # unpermute indices
    Joff = _apply_offset(J, C.offsets)
    Joff, state
end

# Get i-th Cartesian index in memory (permuted) order.
# Returns the Cartesian index in logical (unpermuted) order.
@inline function Base.getindex(
        C::PermutedCartesianIndices, i::Integer)
    @boundscheck checkbounds(C.data, i)
    @inbounds I = C.data[i]  # convert linear to Cartesian index (relatively slow...)
    J = permute_indices(I, C.iperm)  # unpermute indices
    Joff = _apply_offset(J, C.offsets)
    Joff
end

Base.CartesianIndices(A::PencilArray) =
    PermutedCartesianIndices(CartesianIndices(parent(A)), get_permutation(A))

function Base.CartesianIndices(g::GlobalPencilArray)
    A = parent(g)
    off = g.offsets .* -1  # negative offsets
    PermutedCartesianIndices(CartesianIndices(parent(A)), get_permutation(A),
                             off)
end
