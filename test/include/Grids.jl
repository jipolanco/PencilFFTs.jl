module Grids

export PhysicalGrid, FourierGrid
export LocalGridIterator, PhysicalGridIterator, FourierGridIterator
import Base: @propagate_inbounds

using PencilFFTs.PencilArrays
const PA = PencilArrays

using AbstractFFTs: Frequencies, fftfreq, rfftfreq

# N-dimensional grid.
abstract type AbstractGrid{T, N, Perm} end

# Note: the PhysicalGrid is accessed with non-permuted indices.
struct PhysicalGrid{T, N, Perm} <: AbstractGrid{T, N, Perm}
    dims  :: Dims{N}                 # permuted dimensions (N1, N2, N3)
    r     :: NTuple{N, LinRange{T}}  # non-permuted coordinates (x, y, z)
    iperm :: Perm  # inverse permutation (i_2, i_3, i_1) -> (i_1, i_2, i_3)

    # limits: non-permuted geometry limits ((xbegin_1, xend_1), (xbegin_2, xend_2), ...)
    # dims_in: non-permuted global dimensions
    # perm: index permutation
    function PhysicalGrid(limits::NTuple{N, NTuple{2}}, dims_in::Dims{N},
                          perm, ::Type{T}=Float64) where {T, N}
        r = map(limits, dims_in) do l, d
            # Note: we store one extra value at the end (not included in
            # `dims`), to include the right limit.
            LinRange{T}(first(l), last(l), d + 1)
        end
        dims = dims_in
        iperm = inv(perm)
        Perm = typeof(iperm)
        new{T,N,Perm}(dims, r, iperm)
    end
end

# Grid of Fourier wavenumbers for N-dimensional r2c FFTs.
struct FourierGrid{T, N, Perm} <: AbstractGrid{T, N, Perm}
    dims  :: Dims{N}
    r     :: NTuple{N, Frequencies{T}}
    iperm :: Perm

    function FourierGrid(limits::NTuple{N, NTuple{2}}, dims_in::Dims{N},
                         perm, ::Type{T}=Float64) where {T, N}
        F = Frequencies{T}
        r = ntuple(Val(N)) do n
            l = limits[n]
            L = last(l) - first(l)  # box size
            M = dims_in[n]
            fs::T = 2pi * M / L
            n == 1 ? rfftfreq(M, fs)::F : fftfreq(M, fs)::F
        end
        dims = dims_in
        iperm = inv(perm)
        Perm = typeof(iperm)
        new{T,N,Perm}(dims, r, iperm)
    end
end

Base.eltype(::Type{<:AbstractGrid{T}}) where {T} = T
Base.ndims(g::AbstractGrid{T, N}) where {T, N} = N
Base.size(g::AbstractGrid) = g.dims
Base.size(g::AbstractGrid, i) = size(g)[i]
Base.axes(g::AbstractGrid) = Base.OneTo.(size(g))
Base.CartesianIndices(g::AbstractGrid) = CartesianIndices(axes(g))

function Base.iterate(g::AbstractGrid, state::Int=1)
    state_new = state == ndims(g) ? nothing : state + 1
    g[state], state_new
end
Base.iterate(::AbstractGrid, ::Nothing) = nothing

@propagate_inbounds Base.getindex(g::AbstractGrid, i::Integer) = g.r[i]

@propagate_inbounds function Base.getindex(g::AbstractGrid{T, N} where T,
                                           I::CartesianIndex{N}) where N
    # Assume input indices are not permuted.
    t = Tuple(I)
    ntuple(n -> g[n][t[n]], Val(N))
end

@propagate_inbounds function Base.getindex(g::AbstractGrid{T, N} where T,
                                           ranges::NTuple{N, AbstractRange}) where N
    ntuple(n -> g[n][ranges[n]], Val(N))
end

# Get range of geometry associated to a given pencil.
@propagate_inbounds Base.getindex(g::AbstractGrid{T, N} where T,
                                  p::Pencil{N}) where N =
    g[range_local(p, LogicalOrder())]

"""
    LocalGridIterator{T, N, G<:AbstractGrid}

Iterator for efficient access to a subregion of a global grid defined by an
`AbstractGrid` object.
"""
struct LocalGridIterator{
                T,
                N,
                G <: AbstractGrid{T, N},
                It <: Iterators.ProductIterator{<:Tuple{Vararg{AbstractVector,N}}},
                Perm,
            }
    grid  :: G
    range :: NTuple{N, UnitRange{Int}}
    iter  :: It    # iterator with permuted indices and values
    iperm :: Perm  # inverse permutation

    # Note: the range is expected to be unpermuted.
    function LocalGridIterator(grid::AbstractGrid{T,N},
                               range::NTuple{N,UnitRange{Int}}) where {T, N}
        if !(CartesianIndices(range) ⊆ CartesianIndices(grid))
            throw(ArgumentError("given range $range is not a subrange " *
                                "of grid with unpermuted axes = $(axes(grid))"))
        end

        iperm = grid.iperm
        perm = inv(iperm)

        # Note: grid[range] returns non-permuted coordinates from a non-permuted
        # `range`.
        # We want the coordinates permuted. This way we can iterate in the
        # right memory order, according to the current dimension permutation.
        # Then, we unpermute the coordinates at each call to `iterate`.
        grid_perm = perm * grid[range]
        iter = Iterators.product(grid_perm...)

        G = typeof(grid)
        It = typeof(iter)
        Perm = typeof(iperm)

        new{T, N, G, It, Perm}(grid, range, iter, iperm)
    end
end

LocalGridIterator(grid::AbstractGrid, u::PA.MaybePencilArrayCollection) =
    LocalGridIterator(grid, pencil(u))

LocalGridIterator(grid::AbstractGrid, p::Pencil) =
    LocalGridIterator(grid, range_local(p, LogicalOrder()))

Base.parent(g::LocalGridIterator) = g.grid
Base.size(g::LocalGridIterator) = size(g.iter)
Base.eltype(::Type{G} where G <: LocalGridIterator{T}) where {T} = T

@inline function Base.iterate(g::LocalGridIterator, state...)
    next = iterate(g.iter, state...)
    next === nothing && return nothing
    coords_perm, state_new = next  # `coords_perm` is permuted, e.g. (z, y, x)
    # We return unpermuted coordinates, e.g. (x, y, z)
    g.iperm * coords_perm, state_new
end

const FourierGridIterator{T, N} =
    LocalGridIterator{T, N, G} where {T, N, G <: FourierGrid}

const PhysicalGridIterator{T, N} =
    LocalGridIterator{T, N, G} where {T, N, G <: PhysicalGrid}

end
