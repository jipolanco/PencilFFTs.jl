# Get ranks of Cartesian communicator arranged in a matrix representing the
# Cartesian topology.
function get_cart_ranks_matrix(comm::MPI.Comm)
    Ndims = MPI_Cartdim_get(comm)
    @assert Ndims == 2  # only 2D topology is supported

    dims_array, _, _ = MPI_Cart_get(comm, Ndims)
    dims = (dims_array[1], dims_array[2])

    ranks = Matrix{Int}(undef, dims)
    coords = Vector{Cint}(undef, 2)

    for I in CartesianIndices(dims)
        coords .= Tuple(I) .- 1  # MPI uses zero-based indexing
        ranks[I] = MPI_Cart_rank(comm, coords)
    end

    ranks
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
