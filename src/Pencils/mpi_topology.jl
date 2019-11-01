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
