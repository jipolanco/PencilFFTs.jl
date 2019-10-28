"""
    transpose!(dest::PencilArray{T,N}, src::PencilArray{T,N})

Transpose data from one pencil configuration to the other.

The two pencil configurations must be compatible for transposition:

- they must share the same MPI Cartesian topology,

- they must have the same global data size,

- when written as a sorted tuple, the decomposed dimensions must be almost the
  same, with exactly one difference. For instance, if the input of a 3D dataset
  is decomposed in `(2, 3)`, then the output may be decomposed in `(1, 3)`, but
  not in `(1, 2)`.

"""
function transpose!(dest::PencilArray{T,N}, src::PencilArray{T,N}) where {T, N}
    # Verifications
    if src.extra_dims !== dest.extra_dims
        throw(ArgumentError(
            "Incompatible number of extra dimensions of PencilArrays: " *
            "$(src.extra_dims) != $(dest.extra_dims)"))
    end
    assert_compatible(src.pencil, dest.pencil)

    # Note: the `decomp_dims` tuples of both pencils must differ by at most one
    # value (as just checked by `assert_compatible`). The transposition is
    # performed along the dimension R where that difference happens.
    R = findfirst(src.pencil.decomp_dims .!= dest.pencil.decomp_dims)

    if R === nothing
        # Both pencil configurations are identical, so we just copy the data.
        # Actually, this case is currently forbidden by `assert_compatible`.
        return copy!(dest, src)
    end

    transpose_impl!(R, dest, src)
end

function assert_compatible(p::Pencil, q::Pencil)
    if p.topology !== q.topology
        throw(ArgumentError("Pencil topologies must be the same."))
    end
    if p.size_global !== q.size_global
        throw(ArgumentError(
            "Global data sizes must be the same between different pencil " *
            " configurations. Got $(p.size_global) ≠ $(q.size_global)."))
    end
    # Check that decomp_dims differ on exactly one value.
    if sum(p.decomp_dims .!= q.decomp_dims) != 1
        throw(ArgumentError(
            "Pencil decompositions must differ in exactly one dimension. " *
            "Got decomposed dimensions $(p.decomp_dims) and $(q.decomp_dims)."))
    end
    nothing
end

# R: index of MPI subgroup (dimension of MPI Cartesian topology) along which the
# transposition is performed.
function transpose_impl!(R::Int, out::PencilArray{T,N},
                         in::PencilArray{T,N}) where {T, N}
    Pi = in.pencil
    Po = out.pencil
    @assert Pi.topology === Po.topology
    topology = Pi.topology
    comm = topology.subcomms[R]  # exchange among the subgroup R
    Nproc = topology.dims[R]
    myrank = topology.subcomm_ranks[R][topology.coords_local[R]]
    @assert myrank == MPI.Comm_rank(comm)
    @assert Nproc == MPI.Comm_size(comm)
    @assert MPI.Comm_rank(comm) < Nproc

    # Cartesian indices of the remote MPI processes included in the subgroup.
    # Example: if coords_local = (2, 3) and R = 1, then remote_inds holds the
    # indices in (:, 3).
    TopologyIndex = CartesianIndex{ndims(topology)}
    remote_inds = Vector{TopologyIndex}(undef, Nproc)
    let coords = collect(topology.coords_local)  # convert tuple to array
        for n in eachindex(remote_inds)
            coords[R] = n
            remote_inds[n] = TopologyIndex(coords...)
        end
    end

    idims_local = Pi.axes_local
    odims_local = Po.axes_local

    idims = Pi.axes_all
    odims = Po.axes_all

    # TODO
    # - linear buffers
    # - avoid allocations and copies
    # - use @inbounds and @simd
    # - compare with MPI.Alltoallv

    # 1. Send and receive data (TODO avoid allocations...)
    send = Vector{Array{T,N}}(undef, Nproc)
    send_req = Vector{MPI.Request}(undef, Nproc)

    recv = similar(send)
    recv_req = similar(send_req)

    index_local_req = -1  # request index associated to local exchange

    for (n, ind) in enumerate(remote_inds)
        # Global data range that I need to send to process n.
        srange = intersect.(idims_local, odims[ind])

        # Dimensions of data that I will receive from process n.
        rdims = permute_indices(length.(intersect.(odims_local, idims[ind])), Pi)

        # TODO avoid copy / allocation!!
        # Note: Data is sent and received with the permutation associated to Pi.
        send[n] = in[to_local(Pi, srange)...]

        # Exchange data with the other process (non-blocking operations).
        tag = 42
        rank = topology.subcomm_ranks[R][n]  # actual rank of the other process
        if rank == myrank
            recv[n] = send[n]
            send_req[n] = recv_req[n] = MPI.REQUEST_NULL
            index_local_req = n
        else
            recv[n] = Array{T,N}(undef, rdims...)
            send_req[n] = MPI.Isend(send[n], rank, tag, comm)
            recv_req[n] = MPI.Irecv!(recv[n], rank, tag, comm)
        end
    end

    # 2. Unpack data and perform local transposition.
    # Here we need to know the relative index permutation to go from Pi
    # ordering to Po ordering.
    let perm = relative_permutation(Pi, Po)
        no_perm = is_identity_permutation(perm)

        for m = 1:Nproc
            # TODO
            # - avoid repeated operations...
            # - use more consistent variable names with the code above
            if m == 1
                n = index_local_req  # copy local data first
            else
                n, status = MPI.Waitany!(recv_req)
            end

            # Non-permuted global indices of received data.
            ind = remote_inds[n]
            rrange = intersect.(odims_local, idims[ind])

            # Permuted local indices of output data.
            orange = to_local(Po, rrange)

            src = recv[n]
            dest = @view out[orange...]

            # Copy data to `out`, permuting dimensions if required.
            if no_perm
                copy!(dest, src)
            else
                permutedims!(dest, recv[n], perm)
            end
        end
    end

    # Wait for all our data to be sent before returning.
    MPI.Waitall!(send_req)

    out
end
