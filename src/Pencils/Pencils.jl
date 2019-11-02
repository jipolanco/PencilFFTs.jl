"""
Pencil module for 2D decomposition of 3D domains using MPI.

Handles different "pencil" decomposition configurations and data transpositions
between them.
"""
module Pencils

using MPI

import Base: ndims, size, length
import LinearAlgebra: transpose!

export Pencil, PencilArray, MPITopology
export gather
export get_comm, get_decomposition, get_permutation
export ndims_extra
export size_local, size_global
export transpose!

# Describes the portion of an array held by a given MPI process.
const ArrayRegion{N} = NTuple{N,UnitRange{Int}} where N

# Describes indices of an array as a tuple.
const Indices{N} = NTuple{N,Int} where N

"""
    MPITopology{N}

Describes an N-dimensional Cartesian MPI decomposition topology.

---

    MPITopology(comm::MPI.Comm, pdims::Dims{N}) where N

Create N-dimensional MPI topology information.

The `pdims` tuple specifies the number of MPI processes to put in every
dimension of the topology. The product of its values must be equal to the number
of processes in communicator `comm`.

# Example

```julia
# Divide 2D topology into 4×2 blocks.
comm = MPI.COMM_WORLD
@assert MPI.Comm_size(comm) == 8
topology = MPITopology(comm, (4, 2))
```

---

    MPITopology{N}(comm_cart::MPI.Comm) where N

Create topology information from MPI communicator with Cartesian topology.
The topology must have dimension `N`.

"""
struct MPITopology{N}
    # MPI communicator with Cartesian topology.
    comm :: MPI.Comm

    # Subcommunicators associated to the decomposed directions.
    subcomms :: NTuple{N,MPI.Comm}

    # Number of MPI processes along the decomposed directions.
    dims :: Dims{N}

    # Coordinates of the local process in the Cartesian topology.
    # Indices are >= 1.
    coords_local :: Dims{N}

    # Maps Cartesian coordinates to MPI ranks in the `comm` communicator.
    ranks :: Array{Int,N}

    # Maps Cartesian coordinates to MPI ranks in each of the `subcomms`
    # subcommunicators.
    subcomm_ranks :: NTuple{N,Vector{Int}}

    function MPITopology(comm::MPI.Comm, pdims::Dims{N}) where N
        # Create Cartesian communicator.
        comm_cart = let dims = collect(pdims) :: Vector{Int}
            periods = zeros(Int, N)  # this is not very important...
            reorder = false
            MPI.Cart_create(comm, dims, periods, reorder)
        end
        MPITopology{N}(comm_cart)
    end

    function MPITopology{N}(comm_cart::MPI.Comm) where N
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

        ranks = get_cart_ranks(Val(N), comm_cart)
        @assert ranks[coords_local...] == MPI.Comm_rank(comm_cart)

        subcomm_ranks = get_cart_ranks_subcomm.(subcomms)

        new{N}(comm_cart, subcomms, dims, coords_local, ranks, subcomm_ranks)
    end
end

"""
    ndims(t::MPITopology)

Get dimensionality of Cartesian topology.
"""
ndims(t::MPITopology{N}) where N = N

"""
    size(t::MPITopology)

Get dimensions of Cartesian topology.
"""
size(t::MPITopology) = t.dims

"""
    length(t::MPITopology)

Get total size of Cartesian topology (i.e. total number of MPI processes).
"""
length(t::MPITopology) = prod(size(t))

"""
    get_comm(t::MPITopology)

Get MPI communicator associated to an MPI Cartesian topology.
"""
get_comm(t::MPITopology) = t.comm

const Permutation{N} = NTuple{N,Int}
const OptionalPermutation{N} = Union{Nothing, Permutation{N}} where N

"""
    Pencil{N,M}

Describes the decomposition of an `N`-dimensional Cartesian geometry among MPI
processes along `M` directions (with `M < N`).

---

    Pencil(topology::MPITopology{M}, size_global::Dims{N}, decomp_dims::Dims{M};
           permute::P=nothing) where {N, M, P}

Define the decomposition of an `N`-dimensional geometry along `M` dimensions.

The dimensions of the geometry are given by `size_global = (N1, N2, ...)`.

Data is distributed over the given `M`-dimensional MPI topology (with `M < N`).
The decomposed dimensions are given by `decomp_dims`.

The optional parameter `perm` should be a tuple defining a permutation of the
data indices. This may be useful for performance reasons, since it may be
preferable (e.g. for FFTs) that the data is contiguous along the pencil
direction.

# Examples

Decompose a 3D geometry of global dimensions ``N_x × N_y × N_z = 4×8×12`` along
the second (``y``) and third (``z``) dimensions.
```julia
Pencil(topology, (4, 8, 12), (2, 3))             # data is in (x, y, z) order
Pencil(topology, (4, 8, 12), (2, 3), (3, 2, 1))  # data is in (z, y, x) order
```
In the second case, the actual data is stored in `(z, y, x)` order within
each MPI process.

---

    Pencil(p::Pencil{N,M}, decomp_dims::Dims{M},
           size_global::Dims{N}=size_global(p);
           permute::P=get_permutation(p))

Create new pencil configuration from an existent one.

This constructor allows sharing temporary data buffers between the two pencil
configurations, leading to reduced global memory usage.
"""
struct Pencil{N,  # spatial dimensions
              M,  # MPI topology dimensions (< N)
              P<:OptionalPermutation{N},  # optional index permutation
             }
    # M-dimensional MPI decomposition info (with M < N).
    topology :: MPITopology{M}

    # Global array dimensions (N1, N2, ...).
    # These dimensions are *before* permutation by perm.
    size_global :: Dims{N}

    # Decomposition directions (sorted in increasing order).
    # Example: for x-pencils, this is (2, 3, ..., N).
    decomp_dims :: Dims{M}

    # Part of the array held by every process.
    # These dimensions are *before* permutation by `perm`.
    axes_all :: Array{ArrayRegion{N}, M}

    # Part of the array held by the local process (before permutation).
    axes_local :: ArrayRegion{N}

    # Part of the array held by the local process (after permutation).
    axes_local_perm :: ArrayRegion{N}

    # Optional axes permutation.
    perm :: P

    # Data buffers for transpositions.
    send_buf :: Vector{UInt8}
    recv_buf :: Vector{UInt8}

    function Pencil(topology::MPITopology{M}, size_global::Dims{N},
                    decomp_dims::Dims{M};
                    permute::P=nothing,
                    send_buf=UInt8[], recv_buf=UInt8[],
                   ) where {N, M, P<:OptionalPermutation{N}}
        if !is_valid_permuation(permute)
            # This is almost the same error thrown by `permutedims`.
            throw(ArgumentError("invalid permutation of dimensions: $permute"))
        end
        _check_selected_dimensions(N, decomp_dims)
        decomp_dims = _sort_dimensions(decomp_dims)
        axes_all = get_axes_matrix(decomp_dims, topology.dims, size_global)
        axes_local = axes_all[topology.coords_local...]
        axes_local_perm = permute_indices(axes_local, permute)
        new{N,M,P}(topology, size_global, decomp_dims, axes_all, axes_local,
                   axes_local_perm, permute, send_buf, recv_buf)
    end

    function Pencil(p::Pencil{N,M}, decomp_dims::Dims{M},
                    size_global::Dims{N}=size_global(p);
                    permute::P=get_permutation(p),
                   ) where {N, M, P<:OptionalPermutation{N}}
        Pencil(p.topology, size_global, decomp_dims, permute=permute,
               send_buf=p.send_buf, recv_buf=p.recv_buf)
    end
end

# Verify that `dims` is a subselection of dimensions in 1:N.
function _check_selected_dimensions(N, dims::Dims{M}) where M
    if M >= N
        throw(ArgumentError(
            "number of decomposed dimensions `M` must be less than the " *
            "total number of dimensions N = $N (got M = $M)"))
    end
    if length(unique(dims)) != M
        throw(ArgumentError("dimensions may not be repeated. Got $dims."))
    end
    if !all(1 .<= dims .<= N)
        throw(ArgumentError("dimensions must be in 1:$N. Got $dims."))
    end
    nothing
end

function _sort_dimensions(dims::Dims{N}) where N
    s = sort(collect(dims))
    ntuple(n -> s[n], Val(N))  # convert array to tuple
end

"""
    ndims(p::Pencil)

Number of spatial dimensions associated to pencil data.

This corresponds to the total number of dimensions of the space, which includes
the decomposed and non-decomposed dimensions.
"""
ndims(::Pencil{N}) where N = N

"""
    get_comm(p::Pencil)

Get MPI communicator associated to an MPI decomposition scheme.
"""
get_comm(p::Pencil) = get_comm(p.topology)

include("arrays.jl")
include("data_ranges.jl")
include("mpi_topology.jl")
include("mpi_wrappers.jl")
include("permutations.jl")
include("transpose.jl")

"""
    get_permutation(p::Pencil)

Get index permutation associated to the given pencil configuration.

Returns `nothing` if there is no associated permutation.
"""
get_permutation(p::Pencil) = p.perm

"""
    get_decomposition(p::Pencil)

Get tuple with decomposed dimensions of the given pencil configuration.
"""
get_decomposition(p::Pencil) = p.decomp_dims

"""
    size_local(p::Pencil; permute=true)

Local dimensions of the Cartesian grid held by the pencil.

By default the dimensions are permuted to match those of the associated data
arrays.
"""
size_local(p::Pencil{N}; permute::Bool=true) where N =
    length.(permute ? p.axes_local_perm : p.axes_local) :: Dims{N}

"""
    size_global(p::Pencil)

Global dimensions of the Cartesian grid associated to the given domain
decomposition.

Unlike `size_local`, the returned dimensions are *not* permuted to match the
dimensions of the local data.
"""
size_global(p::Pencil) = p.size_global

# Dimensions of remote data for a single process.
# TODO do I need this?
size_remote(p::Pencil{N,M} where N, coords::Vararg{Int,M};
            permute=p.perm) where M =
    permute_indices(length.(p.axes_all[coords...]), permute)

# Dimensions (Nx, Ny, Nz) of remote data for multiple processes.
# TODO do I need this?
function size_remote(p::Pencil{N,M} where N,
                     coords::Vararg{Union{Int,Colon},M};
                     permute=p.perm) where M
    # Returns an array with as many dimensions as colons in `dims`.
    axes = p.axes_all[coords...]
    [permute_indices(length.(ax), permute) for ax in axes]
end

"""
    to_local(p::Pencil, global_inds; permute=true)

Convert non-permuted global indices to local indices.

Indices are permuted by default using the permutation associated to the pencil
configuration `p`.
"""
function to_local(p::Pencil{N}, global_inds::Indices{N};
                  permute=true) where N
    ind = global_inds .- first.(p.axes_local) .+ 1
    permute ? permute_indices(ind, p.perm) : ind
end

function to_local(p::Pencil{N}, global_inds::ArrayRegion{N};
                  permute=true) where N
    ind = map(global_inds, p.axes_local) do rg, rl
        @assert step(rg) == 1
        δ = 1 - first(rl)
        (first(rg) + δ):(last(rg) + δ)
    end :: ArrayRegion{N}
    permute ? permute_indices(ind, p.perm) : ind
end

end
