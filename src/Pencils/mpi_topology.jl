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

## MPI wrappers ##

# TODO add them to MPI.jl...

function MPI_Cart_get!(comm::MPI.Comm, maxdims::Integer,
                       dims::MPI.MPIBuffertype{Cint},
                       periods::MPI.MPIBuffertype{Cint},
                       coords::MPI.MPIBuffertype{Cint})
    # int MPI_Cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[],
    #                  int coords[])
    MPI.@mpichk ccall((:MPI_Cart_get, MPI.libmpi), Cint,
                      (MPI.MPI_Comm, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                      comm, maxdims, dims, periods, coords)
end

function MPI_Cart_get(comm::MPI.Comm, maxdims::Integer)
    dims = Vector{Cint}(undef, maxdims)
    periods = Vector{Cint}(undef, maxdims)
    coords = Vector{Cint}(undef, maxdims)
    MPI_Cart_get!(comm, maxdims, dims, periods, coords)
    Int.(dims), Int.(periods), Int.(coords)
end

function MPI_Cartdim_get(comm::MPI.Comm)
    ndims = Ref{Cint}()
    # int MPI_Cartdim_get(MPI_Comm comm, int *ndims)
    MPI.@mpichk ccall((:MPI_Cartdim_get, MPI.libmpi), Cint,
                      (MPI.MPI_Comm, Ptr{Cint}), comm, ndims)
    Int(ndims[])
end

function MPI_Cart_rank(comm::MPI.Comm, coords::MPI.MPIBuffertype{Cint})
    rank = Ref{Cint}()
    # int MPI_Cart_rank(MPI_Comm comm, const int coords[], int *rank)
    MPI.@mpichk ccall((:MPI_Cart_rank, MPI.libmpi), Cint,
                      (MPI.MPI_Comm, Ptr{Cint}, Ptr{Cint}),
                      comm, coords, rank)
    Int(rank[])
end

function MPI_Cart_rank(comm::MPI.Comm,
                       coords::AbstractArray{T}) where T <: Integer
    ccoords = Cint.(coords[:])
    MPI_Cart_rank(comm, ccoords)
end

function MPI_Cart_sub(comm::MPI.Comm, remain_dims::MPI.MPIBuffertype{Cint})
    newcomm = MPI.Comm()
    # int MPI_Cart_sub(MPI_Comm comm, const int remain_dims[], MPI_Comm *newcomm)
    MPI.@mpichk ccall((:MPI_Cart_sub, MPI.libmpi), Cint,
                      (MPI.MPI_Comm, Ptr{Cint}, Ptr{MPI.MPI_Comm}),
                      comm, remain_dims, newcomm)
    if newcomm.val != MPI.MPI_COMM_NULL
        MPI.refcount_inc()
        finalizer(MPI.free, newcomm)
    end
    newcomm
end

function MPI_Cart_sub(comm::MPI.Comm,
                      remain_dims::AbstractArray{T}) where T <: Integer
    cremain_dims = Cint.(remain_dims[:])
    MPI_Cart_sub(comm, cremain_dims)
end
