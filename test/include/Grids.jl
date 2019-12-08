module Grids

export PhysicalGrid, FourierGrid
export LocalGrid, LocalPhysicalGrid, LocalFourierGrid
import Base: @propagate_inbounds

using PencilFFTs.Pencils

using AbstractFFTs: Frequencies, fftfreq, rfftfreq

# N-dimensional grid.
abstract type AbstractGrid{T, N, Perm} end

# Note: the PhysicalGrid is accessed with permuted indices, and returns non-permuted
# values.
# For example, if the permutation is (2, 3, 1), the PhysicalGrid is accessed with
# indices (i_2, i_3, i_1), and returns values (x_1, x_2, x_3).
struct PhysicalGrid{T, N, Perm} <: AbstractGrid{T, N, Perm}
    dims  :: Dims{N}
    r     :: NTuple{N, LinRange{T}}  # non-permuted coordinates
    iperm :: Perm  # inverse permutation (i_2, i_3, i_1) -> (i_1, i_2, i_3)

    # limits: non-permuted geometry limits ((xbegin_1, xend_1), (xbegin_2, xend_2), ...)
    # dims_in: non-permuted global dimensions
    # perm: index permutation
    function PhysicalGrid(limits::NTuple{N, NTuple{2}}, dims_in::Dims{N},
                  perm, ::Type{T}=Float64) where {T, N}
        r = map(limits, dims_in) do l, d
            LinRange{T}(first(l), last(l), d + 1)
        end
        dims = length.(r)
        iperm = Pencils.inverse_permutation(perm)
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
        dims = length.(r)
        iperm = Pencils.inverse_permutation(perm)
        Perm = typeof(iperm)
        new{T,N,Perm}(dims, r, iperm)
    end
end

Base.eltype(::Type{<:AbstractGrid{T}}) where {T} = T
Base.ndims(g::AbstractGrid{T, N}) where {T, N} = N
Base.size(g::AbstractGrid) = g.dims

function Base.iterate(g::AbstractGrid, state::Int=1)
    state_new = state == ndims(g) ? nothing : state + 1
    g[state], state_new
end
Base.iterate(::AbstractGrid, ::Nothing) = nothing

@propagate_inbounds Base.getindex(g::AbstractGrid, i::Integer) = g.r[i]

@propagate_inbounds function Base.getindex(g::AbstractGrid{T, N} where T,
                                           I::CartesianIndex{N}) where N
    # Assume input indices are permuted, and un-permute them.
    t = Pencils.permute_indices(Tuple(I), g.iperm)
    ntuple(n -> g[n][t[n]], Val(N))
end

@propagate_inbounds function Base.getindex(g::AbstractGrid{T, N} where T,
                                           ranges::NTuple{N, AbstractRange}) where N
    # Assume input indices are permuted, and un-permute them.
    t = Pencils.permute_indices(ranges, g.iperm)
    ntuple(n -> g[n][t[n]], Val(N))
end

# Get range of geometry associated to a given pencil.
@propagate_inbounds Base.getindex(g::AbstractGrid{T, N} where T,
                                  p::Pencil{N}) where N =
    g[range_local(p, permute=true)]

"""
    LocalGrid{T, N, G<:AbstractGrid} <: AbstractArray{T, N}

Allows access to a subregion of a global grid defined by an `AbstractGrid`
object.

As opposed to `AbstractGrid`s, which compute values lazily, a `LocalGrid`
stores all the values of the local grid in N-dimensional `Array`s. This
enables efficient access to grid values using linear indexing, but costs a bit
of memory (same as a local vector field).

Also note that a `LocalGrid` takes local indices (starting from 1).
"""
struct LocalGrid{T,
                 N,
                 G <: AbstractGrid{T, N},
                } <: AbstractArray{T, N}
    grid  :: G
    range :: NTuple{N, UnitRange{Int}}
    data  :: NTuple{N, Array{T, N}}
    dims  :: Dims{N}

    # Note: the range is expected to be permuted (just like the indices that are
    # passed to AbstractGrid).
    function LocalGrid(grid::AbstractGrid{T,N},
                       range::NTuple{N,UnitRange{Int}}) where {T, N}
        # TODO verify that `range` is a subrange of `grid`
        dims = length.(range)
        data = ntuple(n -> Array{T}(undef, dims), Val(N))
        for (i, I) in enumerate(CartesianIndices(range))
            g = grid[I]
            for n = 1:N
                data[n][i] = g[n]
            end
        end
        G = typeof(grid)
        new{T, N, G}(grid, range, data, dims)
    end
end

LocalGrid(grid::AbstractGrid,
          u::Pencils.MaybePencilArrayCollection) = LocalGrid(grid, pencil(u))
LocalGrid(grid::AbstractGrid, p::Pencil) =
    LocalGrid(grid, range_local(p, permute=true))

Base.size(g::LocalGrid) = g.dims

Base.IndexStyle(::Type{<:LocalGrid}) = IndexLinear()

const LocalFourierGrid{T, N} = LocalGrid{T, N, G} where {T, N, G <: FourierGrid}
const LocalPhysicalGrid{T, N} = LocalGrid{T, N, G} where {T, N, G <: PhysicalGrid}

@propagate_inbounds function Base.getindex(g::LocalGrid, inds...)
    # Since all the arrays in `data` have the same indices, we explicitly
    # convert to linear indices only once. This only makes a difference if more
    # than one index (or a CartesianIndex) was passed as input.
    i = LinearIndices(first(g.data))[inds...]
    map(x -> x[i], g.data)
end

end
