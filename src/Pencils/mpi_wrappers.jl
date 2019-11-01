# Extra MPI C wrappers that are not currently included in MPI.jl.
# The wrappers are written in the same style as in MPI.jl, and they should
# probably be added to that package in the future...

## These extend MPI.jl/src/topology.jl

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

## These extend MPI.jl/src/pointtopoint.jl
## (But I'm not sure if they should be added to MPI.jl...)

# Alternative wrapper to Waitany that takes a Vector{MPI.MPI_Request} instead of
# a Vector{MPI.Request} (the former is a C integer, while the latter is a MPI.jl
# wrapper that contains more information). This avoids the allocation of a
# vector every time Waitany! is called.
function MPI_Waitany!(reqvals::Vector{MPI.MPI_Request})
    count = length(reqvals)
    ind = Ref{Cint}()
    stat_ref = Ref{MPI.Status}()
    # int MPI_Waitany(int count, MPI_Request array_of_requests[], int *index,
    #                 MPI_Status *status)
    MPI.@mpichk ccall((:MPI_Waitany, MPI.libmpi), Cint,
                      (Cint, Ptr{MPI.MPI_Request}, Ptr{Cint}, Ptr{MPI.Status}),
                      count, reqvals, ind, stat_ref)
    index = Int(ind[]) + 1
    (index, stat_ref[])
end

# Same for Waitall.
function MPI_Waitall!(reqvals::Vector{MPI.MPI_Request})
    count = length(reqvals)
    stats = Array{MPI.Status}(undef, count)
    # int MPI_Waitall(int count, MPI_Request array_of_requests[],
    #                 MPI_Status array_of_statuses[])
    MPI.@mpichk ccall((:MPI_Waitall, MPI.libmpi), Cint,
                      (Cint, Ptr{MPI.MPI_Request}, Ptr{MPI.Status}),
                      count, reqvals, stats)
    stats
end
