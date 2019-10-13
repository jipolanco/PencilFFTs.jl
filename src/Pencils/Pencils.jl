"""
Pencil module for 2D decomposition of 3D domains using MPI.

Handles different "pencil" decomposition configurations and data transpositions
between them.
"""
module Pencils

using MPI

export Pencil

include("data_ranges.jl")
include("mpi_topology.jl")

# Describes the portion of an array held by a given MPI process.
const ArrayRegion{N} = NTuple{N,UnitRange{Int}} where N

# TODO
# - create sub-communicators for each decomposition direction?

"""
    Pencil{D}

Describes the decomposition of a 3D array in a single pencil decomposition
configuration.

The pencil is oriented in the direction `D` (with `D ∈ 1:3`).

---

    Pencil{D}(comm_cart::MPI.Comm, size_global) where D

Define pencil decomposition along direction `D` for an array of dimensions `size_global = (Nx, Ny, Nz)`.

The MPI communicator `comm_cart` must have a 2D Cartesian topology.
This kind of communicator is usually obtained from `MPI.Cart_create`.

---

    Pencil{D}(p::Pencil{S}) where {D, S}

Create new pencil configuration from an existent one.

The new pencil is constructed in a way that enables efficient data
transpositions between the two configurations.

"""
struct Pencil{D}
    # MPI communicator.
    comm :: MPI.Comm

    # Number of MPI processes along the decomposed directions.
    cart_dims :: Dims{2}

    # Coordinates of the local pencil in the Cartesian topology.
    # Indices are >= 1.
    cart_coords_local :: Dims{2}

    # Maps Cartesian coordinates of pencils to MPI ranks.
    cart_ranks :: Matrix{Int}

    # Global array dimensions (Nx, Ny, Nz).
    size_global :: Dims{3}

    # Part of the array held by every process.
    axes_all :: Matrix{ArrayRegion{3}}

    # Part of the array held by the local process.
    axes_local :: ArrayRegion{3}

    function Pencil{D}(comm_cart::MPI.Comm, size_global::NTuple{3}) where D
        # Get dimensions of MPI topology.
        # This will fail if comm_cart doesn't have Cartesian topology!
        Ndims = MPI_Cartdim_get(comm_cart)

        if Ndims != 2
            throw(ArgumentError(
                "Cartesian communicator must have two dimensions."))
        end

        cart_dims, cart_coords_local = let maxdims = 2
            dims, _, coords = MPI_Cart_get(comm_cart, maxdims)
            coords .+= 1  # switch to one-based indexing
            (dims[1], dims[2]), (coords[1], coords[2])
        end

        cart_ranks = get_cart_ranks_matrix(comm_cart)
        @assert cart_ranks[cart_coords_local...] == MPI.Comm_rank(comm_cart)

        axes_all = get_axes_matrix(Val(D), cart_dims, size_global)
        axes_local = axes_all[cart_coords_local...]

        new{D}(comm_cart, cart_dims, cart_coords_local, cart_ranks,
               size_global, axes_all, axes_local)
    end

    # Case S = D (not very useful...)
    Pencil{D}(p::Pencil{D}) where {D} = p

    # General case S != D
    function Pencil{D}(p::Pencil{S}) where {D, S}
        axes_all = get_axes_matrix(Val(D), p.cart_dims, p.size_global)
        axes_local = axes_all[p.cart_coords_local...]
        new(p.comm, p.cart_dims, p.cart_coords_local, p.cart_ranks,
            p.size_global, axes_all, axes_local)
    end
end

end
