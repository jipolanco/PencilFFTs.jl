"""
Pencil module for 2D decomposition of 3D domains using MPI.

Handles different "pencil" decomposition configurations and data transpositions
between them.
"""
module Pencils

using MPI

export Pencil
export allocate
export transpose!

# Describes the portion of an array held by a given MPI process.
const ArrayRegion{N} = NTuple{N,UnitRange{Int}} where N

# TODO
# - define PencilArray array wrappers containing data + pencil info
# - define PencilArray data allocators from one or more pencils.
#   The returned array must be large enough to fit data from all pencils.

"""
    Topology{N}

Describes an N-dimensional Cartesian MPI decomposition topology.
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

    # Maps Cartesian coordinates to MPI ranks.
    ranks :: Matrix{Int}

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

        ranks = get_cart_ranks_matrix(Val(N), comm_cart)
        @assert ranks[coords_local...] == MPI.Comm_rank(comm_cart)

        subcomms = create_subcomms(Val(N), comm_cart)
        @assert MPI.Comm_size.(subcomms) === dims

        new{N}(comm_cart, subcomms, dims, coords_local, ranks)
    end
end

"""
    Pencil{D}

Describes the decomposition of a 3D array in a single pencil decomposition
configuration.

The pencil is oriented in the direction `D` (with `D ∈ 1:3`).

---

    Pencil{D}(comm_cart::MPI.Comm, size_global) where D

Define pencil decomposition along direction `D` for an array of dimensions
`size_global = (Nx, Ny, Nz)`.

The MPI communicator `comm_cart` must have a 2D Cartesian topology.
This kind of communicator is usually obtained from `MPI.Cart_create`.

---

    Pencil{D}(p::Pencil{S}) where {D, S}

Create new pencil configuration from an existent one.

The new pencil is constructed in a way that enables efficient data
transpositions between the two configurations.

"""
struct Pencil{D}
    # Two-dimensional MPI decomposition info.
    # This should be the same for all pencil configurations.
    topology :: Topology{2}

    # Global array dimensions (Nx, Ny, Nz).
    size_global :: Dims{3}

    # Part of the array held by every process.
    axes_all :: Matrix{ArrayRegion{3}}

    # Part of the array held by the local process.
    axes_local :: ArrayRegion{3}

    function Pencil{D}(comm_cart::MPI.Comm, size_global::Dims{3}) where D
        topology = Topology{2}(comm_cart)
        axes_all = get_axes_matrix(Val(D), topology.dims, size_global)
        axes_local = axes_all[topology.coords_local...]
        new{D}(topology, size_global, axes_all, axes_local)
    end

    # Case S = D (not very useful...)
    Pencil{D}(p::Pencil{D}) where {D} = p

    # General case S != D
    function Pencil{D}(p::Pencil{S}) where {D, S}
        axes_all = get_axes_matrix(Val(D), p.topology.dims, p.size_global)
        axes_local = axes_all[p.topology.coords_local...]
        new(p.topology, p.size_global, axes_all, axes_local)
    end
end

include("data_ranges.jl")
include("mpi_topology.jl")
include("transpose.jl")

size_local(p::Pencil) = length.(p.axes_local)

"""
    allocate(p::Pencil, [T=Float64])

Allocate uninitialised 3D array with the dimensions of the given pencil.
"""
allocate(p::Pencil, ::Type{T}=Float64) where T = Array{T}(undef, size_local(p))

end
