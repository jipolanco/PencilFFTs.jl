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

function show(io::IO, t::MPITopology)
    M = ndims(t)
    s = join(size(t), '×')
    print(io, "MPI topology: $(M)D decomposition ($s processes)")
    nothing
end

"""
    ndims(t::MPITopology)

Get dimensionality of Cartesian topology.
"""
Base.ndims(t::MPITopology{N}) where N = N

"""
    size(t::MPITopology)

Get dimensions of Cartesian topology.
"""
Base.size(t::MPITopology) = t.dims

"""
    length(t::MPITopology)

Get total size of Cartesian topology (i.e. total number of MPI processes).
"""
Base.length(t::MPITopology) = prod(size(t))

"""
    get_comm(t::MPITopology)

Get MPI communicator associated to an MPI Cartesian topology.
"""
get_comm(t::MPITopology) = t.comm

# Get ranks of N-dimensional Cartesian communicator.
function get_cart_ranks(::Val{N}, comm::MPI.Comm) where N
    @assert MPI_Cartdim_get(comm) == N  # communicator should be N-dimensional
    Nproc = MPI.Comm_size(comm)

    dims = begin
        dims_vec, _, _ = MPI_Cart_get(comm, N)
        ntuple(n -> dims_vec[n], N)
    end

    ranks = Array{Int,N}(undef, dims)
    coords = Vector{Cint}(undef, N)

    for I in CartesianIndices(dims)
        coords .= Tuple(I) .- 1  # MPI uses zero-based indexing
        ranks[I] = MPI_Cart_rank(comm, coords)
    end

    ranks
end

# Get ranks of one-dimensional Cartesian sub-communicator.
function get_cart_ranks_subcomm(subcomm::MPI.Comm)
    @assert MPI_Cartdim_get(subcomm) == 1  # sub-communicator should be 1D
    Nproc = MPI.Comm_size(subcomm)

    ranks = Vector{Int}(undef, Nproc)
    coords = Ref{Cint}()

    for n = 1:Nproc
        coords[] = n - 1  # MPI uses zero-based indexing
        ranks[n] = MPI_Cart_rank(subcomm, coords)
    end

    ranks
end

function create_subcomms(::Val{N}, comm::MPI.Comm) where N
    remain_dims = Vector{Cint}(undef, N)
    ntuple(Val(N)) do n
        fill!(remain_dims, zero(Cint))
        remain_dims[n] = one(Cint)
        MPI_Cart_sub(comm, remain_dims)
    end
end
