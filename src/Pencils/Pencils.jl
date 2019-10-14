"""
Pencil module for 2D decomposition of 3D domains using MPI.

Handles different "pencil" decomposition configurations and data transpositions
between them.
"""
module Pencils

using MPI

import Base: ndims

export Pencil
export allocate, index_permutation
export transpose!

# Describes the portion of an array held by a given MPI process.
const ArrayRegion{N} = NTuple{N,UnitRange{Int}} where N

# Describes indices of an array as a tuple.
const Indices{N} = NTuple{N,Int} where N

# Number of dimensions of Cartesian MPI topology.
const TOPOLOGY_DIMS = 2

# TODO
# - define PencilArray array wrappers containing data + pencil info
# - define PencilArray data allocators from one or more pencils.
#   The returned array must be large enough to fit data from all pencils.

"""
    Topology{N}

Describes an N-dimensional Cartesian MPI decomposition topology.

---

    Topology{N}(comm_cart::MPI.Comm) where N

Create topology information from MPI communicator with Cartesian topology.
The topology must have dimension `N`.

"""
struct Topology{N}
    # MPI communicator with Cartesian topology.
    comm :: MPI.Comm

    # Subcommunicators associated to the two decomposed directions.
    subcomms :: NTuple{N,MPI.Comm}

    # Number of MPI processes along the decomposed directions.
    dims :: Dims{N}

    # Coordinates of the local process in the Cartesian topology.
    # Indices are >= 1.
    coords_local :: Dims{N}

    # Maps Cartesian coordinates to MPI ranks in each of the `subcomms`
    # communicators.
    ranks :: NTuple{N,Vector{Int}}

    function Topology{N}(comm_cart::MPI.Comm) where N
        # Get dimensions of MPI topology.
        # This will fail if comm_cart doesn't have Cartesian topology!
        Ndims = MPI_Cartdim_get(comm_cart)

        if Ndims != N
            throw(ArgumentError(
                "Cartesian communicator must have $N dimensions."))
        end

        dims, coords_local = begin
            dims_vec, _, coords_vec = MPI_Cart_get(comm_cart, N)
            coords_vec .+= 1  # switch to one-based indexing
            map(X -> ntuple(n -> X[n], Val(N)), (dims_vec, coords_vec))
        end

        subcomms = create_subcomms(Val(N), comm_cart)
        @assert MPI.Comm_size.(subcomms) === dims

        ranks = get_cart_ranks_subcomm.(subcomms)
        # @assert ranks[coords_local...] == MPI.Comm_rank(comm_cart)

        new{N}(comm_cart, subcomms, dims, coords_local, ranks)
    end
end

ndims(t::Topology{N}) where N = N

const Permutation{N} = NTuple{N,Int}
const OptionalPermutation{N} = Union{Nothing, Permutation{N}} where N

"""
    Pencil{D}

Describes the decomposition of a 3D array in a single pencil decomposition
configuration.

The pencil is oriented in the direction `D` (with `D ∈ 1:3`), meaning that
MPI decomposition is performed in the other two directions.

---

    Pencil{D}(comm_cart::MPI.Comm, size_global; permute::P=nothing) where {D, P}

Define pencil decomposition along direction `D` for an array of dimensions
`size_global = (Nx, Ny, Nz)`.

The MPI communicator `comm_cart` must have a 2D Cartesian topology.
This kind of communicator is usually obtained from `MPI.Cart_create`.

The optional parameter `perm` should be a tuple defining a permutation of the
data indices. This may be useful for performance reasons, since it may be
preferable (e.g. for FFTs) that the data is contiguous along the pencil
direction.

# Examples

```julia
Pencil{D}(comm, (4, 8, 12))             # data is in (Nx, Ny, Nz) order
Pencil{D}(comm, (4, 8, 12), (3, 2, 1))  # data is in (Nz, Ny, Nx) order
```

---

    Pencil{D}(p::Pencil{S}; permute::P=nothing) where {D, S, P}

Create new pencil configuration from an existent one.

The new pencil is constructed in a way that enables efficient data
transpositions between the two configurations.

"""
struct Pencil{D, P<:OptionalPermutation{3}}
    # Two-dimensional MPI decomposition info.
    # This should be the same for all pencil configurations.
    # TODO generalise to N dimensions (1 <= N <= 3)?
    topology :: Topology{TOPOLOGY_DIMS}

    # Global array dimensions (Nx, Ny, Nz).
    # These dimensions are *before* permutation by perm.
    size_global :: Dims{3}

    # Part of the array held by every process.
    # These dimensions are *before* permutation by perm.
    axes_all :: Array{ArrayRegion{3}, TOPOLOGY_DIMS}

    # Part of the array held by the local process.
    axes_local :: ArrayRegion{3}

    # Optional axes permutation.
    perm :: P

    function Pencil{D}(comm_cart::MPI.Comm, size_global::Dims{3};
                       permute::P=nothing) where {D, P<:OptionalPermutation{3}}
        topology = Topology{2}(comm_cart)
        axes_all = get_axes_matrix(Val(D), topology.dims, size_global)
        axes_local = axes_all[topology.coords_local...]
        new{D,P}(topology, size_global, axes_all, axes_local, permute)
    end

    function Pencil{D}(p::Pencil{S}; permute::P=nothing) where
            {D, S, P<:OptionalPermutation{3}}
        axes_all = get_axes_matrix(Val(D), p.topology.dims, p.size_global)
        axes_local = axes_all[p.topology.coords_local...]
        new{D,P}(p.topology, p.size_global, axes_all, axes_local, permute)
    end
end

include("data_ranges.jl")
include("mpi_topology.jl")
include("permutations.jl")
include("transpose.jl")

"""
    index_permutation(p::Pencil)

Get index permutation associated to the given pencil.

Returns `nothing` if there is no associated permutation.
"""
index_permutation(p::Pencil) = p.perm

# Dimensions (Nx, Ny, Nz) of local data (possibly permuted).
# Set `permute=nothing` to disable index permutation.
size_local(p::Pencil; permute=p.perm) =
    permute_indices(length.(p.axes_local), permute)

# Dimensions of remote data for a single process.
# TODO do I need this?
size_remote(p::Pencil, dims::Vararg{Int,TOPOLOGY_DIMS}; permute=p.perm) =
    permute_indices(length.(p.axes_all[dims...]), permute)

# Dimensions (Nx, Ny, Nz) of remote data for multiple processes.
# TODO do I need this?
function size_remote(p::Pencil, dims::Vararg{Union{Int,Colon},TOPOLOGY_DIMS};
                     permute=p.perm)
    # Returns an array with as many dimensions as colons in `dims`.
    axes = p.axes_all[dims...]
    [permute_indices(length.(ax), permute) for ax in axes]
end

"""
    to_local(p::Pencil, global_inds; permute=p.perm)

Convert non-permuted global indices to local indices, which are permuted by default.
"""
to_local(p::Pencil, global_inds::Indices{3}; permute=p.perm) =
    permute_indices(global_inds .- first.(p.axes_local) .+ 1, permute)

function to_local(p::Pencil, global_inds::ArrayRegion{3}; permute=p.perm)
    ind = map(global_inds, p.axes_local) do rg, rl
        @assert step(rg) == 1
        δ = 1 - first(rl)
        (first(rg) + δ):(last(rg) + δ)
    end :: ArrayRegion{3}
    permute_indices(ind, permute)
end

"""
    allocate(p::Pencil, [T=Float64])

Allocate uninitialised 3D array with the dimensions of the given pencil.

Data is permuted if the pencil was defined with a given permutation.
"""
allocate(p::Pencil, ::Type{T}=Float64) where T = Array{T}(undef, size_local(p))

end
