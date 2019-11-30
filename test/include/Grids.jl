module Grids

export Grid

using PencilFFTs.Pencils

# using AbstractFFTs: fftfreq, rfftfreq

# N-dimensional grid.
abstract type AbstractGrid{N} end

struct Grid{N} <: AbstractGrid{N}
    dims :: Dims{N}
    r :: NTuple{N, LinRange{Float64}}
    function Grid(limits::NTuple{N, NTuple{2}}, dims::Dims{N}) where N
        r = map(limits, dims) do l, d
            LinRange{Float64}(first(l), last(l), d + 1)
        end
        new{N}(dims, r)
    end
end

# Grid of Fourier wavenumbers for N-dimensional r2c FFTs.
struct FourierGrid{N} <: AbstractGrid{N}
    dims :: Dims{N}
    r :: NTuple{N, Vector{Float64}}
    # function FourierGrid()
    #     # TODO...
    # end
end

Base.size(g::AbstractGrid) = g.dims

Base.getindex(g::AbstractGrid, i::Integer) = g.r[i]
Base.getindex(g::AbstractGrid, i::Integer, j) = g[i][j]

function Base.getindex(g::AbstractGrid{N}, I::CartesianIndex{N}) where N
    t = Tuple(I)
    ntuple(n -> g[n, t[n]], Val(N))
end

Base.getindex(g::AbstractGrid{N}, ranges::NTuple{N, AbstractRange}) where N =
    ntuple(n -> g[n, ranges[n]], Val(N))

# Get range of geometry associated to a given pencil.
Base.getindex(g::AbstractGrid{N}, p::Pencil{N}) where N =
    g[range_local(p, permute=false)]

end
