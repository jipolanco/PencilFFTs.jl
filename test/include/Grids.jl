module Grids

export Grid, FourierGrid
import Base: @propagate_inbounds

using PencilFFTs.Pencils

using AbstractFFTs: Frequencies, fftfreq, rfftfreq

# N-dimensional grid.
abstract type AbstractGrid{N, Perm} end

# Note: the Grid is accessed with permuted indices, and returns non-permuted
# values.
# For example, if the permutation is (2, 3, 1), the Grid is accessed with
# indices (i_2, i_3, i_1), and returns values (x_1, x_2, x_3).
struct Grid{N, Perm} <: AbstractGrid{N, Perm}
    dims  :: Dims{N}
    r     :: NTuple{N, LinRange{Float64}}  # non-permuted coordinates
    iperm :: Perm  # inverse permutation (i_2, i_3, i_1) -> (i_1, i_2, i_3)

    # limits: non-permuted geometry limits ((xbegin_1, xend_1), (xbegin_2, xend_2), ...)
    # dims_in: non-permuted global dimensions
    # perm: index permutation
    function Grid(limits::NTuple{N, NTuple{2}}, dims_in::Dims{N},
                  perm) where {N}
        r = map(limits, dims_in) do l, d
            LinRange{Float64}(first(l), last(l), d + 1)
        end
        dims = length.(r)
        iperm = Pencils.inverse_permutation(perm)
        Perm = typeof(iperm)
        new{N,Perm}(dims, r, iperm)
    end
end

# Grid of Fourier wavenumbers for N-dimensional r2c FFTs.
struct FourierGrid{N, Perm} <: AbstractGrid{N, Perm}
    dims  :: Dims{N}
    r     :: NTuple{N, Frequencies{Float64}}
    iperm :: Perm

    function FourierGrid(limits::NTuple{N, NTuple{2}}, dims_in::Dims{N},
                         perm) where {N}
        r = ntuple(Val(N)) do n
            l = limits[n]
            L = last(l) - first(l)  # box size
            M = dims_in[n]
            fs = 2pi * M / L
            n == 1 ? rfftfreq(M, fs) : fftfreq(M, fs)
        end
        dims = length.(r)
        iperm = Pencils.inverse_permutation(perm)
        Perm = typeof(iperm)
        new{N,Perm}(dims, r, iperm)
    end
end

Base.ndims(g::AbstractGrid{N}) where {N} = N
Base.size(g::AbstractGrid) = g.dims

function Base.iterate(g::AbstractGrid, state::Int=1)
    state_new = state == ndims(g) ? nothing : state + 1
    g[state], state_new
end
Base.iterate(::AbstractGrid, ::Nothing) = nothing

@propagate_inbounds Base.getindex(g::AbstractGrid, i::Integer) = g.r[i]

@propagate_inbounds function Base.getindex(g::AbstractGrid{N},
                                           I::CartesianIndex{N}) where N
    # Assume input indices are permuted, and un-permute them.
    t = Pencils.permute_indices(Tuple(I), g.iperm)
    ntuple(n -> g[n][t[n]], Val(N))
end

@propagate_inbounds function Base.getindex(g::AbstractGrid{N},
                                           ranges::NTuple{N, AbstractRange}) where N
    # Assume input indices are permuted, and un-permute them.
    t = Pencils.permute_indices(ranges, g.iperm)
    ntuple(n -> g[n][t[n]], Val(N))
end

# Get range of geometry associated to a given pencil.
@propagate_inbounds Base.getindex(g::AbstractGrid{N}, p::Pencil{N}) where N =
    g[range_local(p, permute=true)]

end
