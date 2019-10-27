"""
    transpose!(out::PencilArray{T,N}, in::PencilArray{T,N})

Transpose data from one pencil configuration to the other.

The two pencil configurations must be compatible for transposition:

- they must share the same MPI Cartesian topology,

- they must have the same global data size,

- when written as a sorted tuple, the decomposed dimensions must be almost the
  same, with exactly one difference. For instance, if the input of a 3D dataset
  is decomposed in `(2, 3)`, then the output may be decomposed in `(1, 3)`, but
  not in `(1, 2)`.

"""
transpose!(out::PencilArray{T,N}, in::PencilArray{T,N}) where {T, N} =
    transpose_impl!(out, out.pencil, in, in.pencil)

function assert_compatible(p::Pencil, q::Pencil)
    if p.topology !== q.topology
        throw(ArgumentError("Pencil topologies must be the same."))
    end
    if p.size_global !== q.size_global
        throw(ArgumentError("Global data sizes must be the same between " *
                            "different pencil configurations."))
    end
    # Check that decomp_dims differ on exactly one value.
    if sum(p.decomp_dims .!= q.decomp_dims) != 1
        throw(ArgumentError(
            "Pencil configurations must differ in exactly one dimension."))
    end
    nothing
end


# TODO generalise to all pencil types
# Transpose Pencil{1} -> Pencil{2}.
# Decomposition dimensions switch from (2, 3) to (1, 3).
function transpose_impl!(out::PencilArray{T,N}, Pout::Pencil{2},
                         in::PencilArray{T,N}, Pin::Pencil{1}) where {T, N}
    @assert in.pencil === Pin
    @assert out.pencil === Pout

    # Verifications
    if in.extra_dims !== out.extra_dims
        throw(ArgumentError(
            "Incompatible number of extra dimensions of PencilArrays: " *
            "$(in.extra_dims) != $(out.extra_dims)"))
    end
    assert_compatible(Pin, Pout)

    # Transposition is performed in the first Cartesian dimension
    # (P1 = 2 -> 1), hence R = 1.
    # TODO what happens if I put Val(2)? Error somewhere?
    @assert Pin.decomp_dims[1] == 2 && Pout.decomp_dims[1] == 1

    # These should be the same (TODO verify!):
    transpose_impl!(1, out, Pout, in, Pin)
    # transpose_impl!(1, out.data, Pout, in.data, Pin)
end

# R: index of MPI subgroup (dimension of MPI Cartesian topology) among which the
# transposition is performed.
function transpose_impl!(R::Int, out::AbstractArray{T,N}, Pout::Pencil{2},
                         in::AbstractArray{T,N}, Pin::Pencil{1}) where {T, N}
    # Pencil{1} -> Pencil{2} transpose **must** be done via subgroup 1
    @assert R == 1
    @assert N == 3  # for now only this is supported

    @assert Pin.topology === Pout.topology
    topology = Pin.topology
    comm = topology.subcomms[R]  # exchange among the subgroup R
    Nproc = topology.dims[R]
    myrank = topology.subcomm_ranks[R][topology.coords_local[R]]
    @assert myrank == MPI.Comm_rank(comm)
    @assert Nproc == MPI.Comm_size(comm)
    @assert MPI.Comm_rank(comm) < Nproc

    idims_local = Pin.axes_local
    odims_local = Pout.axes_local

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

    idims = Pin.axes_all
    odims = Pout.axes_all

    # TODO
    # - avoid allocations and copies
    # - avoid sending data to myself
    # - use @inbounds
    # - compare with MPI.Alltoallv

    # 1. Send and receive data (TODO avoid allocations...)
    send = Vector{Array{T,N}}(undef, Nproc)
    send_req = Vector{MPI.Request}(undef, Nproc)

    recv = similar(send)
    recv_req = similar(send_req)

    index_local_req = -1  # request index associated to local exchange

    for (n, ind) in enumerate(remote_inds)
        # Global data range that I need to send to process n.
        # Intersections must be done with unpermuted indices!
        # Note: Data is sent and received with the permutation associated to Pin.
        srange = permute_indices((intersect.(idims_local, odims[ind])), Pin)

        # Dimensions of data that I will receive from process n.
        rdims = permute_indices(length.(intersect.(odims_local, idims[ind])), Pin)

        # TODO avoid copy / allocation!!
        send[n] = in[to_local(Pin, srange)...]

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
    # Here we need to know the relative index permutation to go from Pin
    # ordering to Pout ordering.
    let perm = relative_permutation(Pin, Pout)
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
            orange = to_local(Pout, rrange)

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
