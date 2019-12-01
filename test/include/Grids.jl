module Grids

export Grid, FourierGrid

using PencilFFTs.Pencils

using AbstractFFTs: Frequencies, fftfreq, rfftfreq

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
    r :: NTuple{N, Frequencies{Float64}}
    function FourierGrid(limits::NTuple{N, NTuple{2}}, dims_in::Dims{N}) where N
        r = ntuple(Val(N)) do n
            l = limits[n]
            L = last(l) - first(l)  # box size
            M = dims_in[n]
            fs = 2pi * M / L
            n == 1 ? rfftfreq(M, fs) : fftfreq(M, fs)
        end
        dims = length.(r)
        new{N}(dims, r)
    end
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
