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

    @inbounds transpose_impl!(R, dest, src)
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

# Reinterpret UInt8 array as a different type of array.
# This is a workaround to the performance issues when using `reinterpret`.
# See for instance:
# - https://discourse.julialang.org/t/big-overhead-with-the-new-lazy-reshape-reinterpret/7635
# - https://github.com/JuliaLang/julia/issues/28980
unsafe_as_array(::Type{T}, x::Vector{UInt8}) where T =
    unsafe_wrap(Array, convert(Ptr{T}, pointer(x)), length(x) ÷ sizeof(T),
                own=false)

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
    # - avoid copies?
    # - use @simd?
    # - compare with MPI.Alltoallv

    # Length of data that I will "send" to myself.
    length_send_local = prod(length.(intersect.(idims_local, odims_local)))

    # Total data to be sent / received.
    length_send = length(in) - length_send_local
    length_recv = length(out)  # includes local exchange with myself

    # 1. Send and receive data.
    # Note: I prefer to resize the original UInt8 array instead of the "unsafe"
    # Array{T}.
    resize!(Po.send_buf, sizeof(T) * length_send)
    send_buf = unsafe_as_array(T, Po.send_buf)
    @assert length(send_buf) == length_send
    send_req = Vector{MPI.Request}(undef, Nproc)
    isend = 0  # current index in send_buf

    resize!(Po.recv_buf, sizeof(T) * length_recv)
    recv_buf = unsafe_as_array(T, Po.recv_buf)
    @assert length(recv_buf) == length_recv
    recv_req = similar(send_req)
    irecv = 0  # current index in recv_buf
    recv_offsets = Vector{Int}(undef, Nproc)  # all offsets in recv_buf

    index_local_req = -1  # request index associated to local exchange

    for (n, ind) in enumerate(remote_inds)
        # Global data range that I need to send to process n.
        srange = intersect.(idims_local, odims[ind]) # Dimensions of data that I will receive from process n.
        rrange = intersect.(odims_local, idims[ind])
        rdims = permute_indices(length.(rrange), Pi)

        length_recv_n = prod(rdims)
        recv_offsets[n] = irecv
        irecv_prev = irecv
        irecv += length_recv_n
        recv_buf_view = @view recv_buf[(irecv_prev + 1):irecv]

        # Note: data is sent and received with the permutation associated to Pi.
        to_send = @view in[to_local(Pi, srange)...]

        # Exchange data with the other process (non-blocking operations).
        tag = 42
        rank = topology.subcomm_ranks[R][n]  # actual rank of the other process
        if rank == myrank
            # Copy directly to_send -> recv_buf_view.
            @assert length(recv_buf_view) == length(to_send)
            copyto!(recv_buf_view, to_send)
            send_req[n] = recv_req[n] = MPI.REQUEST_NULL
            index_local_req = n
        else
            # Copy to_send into contiguous buffer, then send the buffer.
            # TODO If to_send is contiguous, avoid copying data to buffer.
            # (I need to check if it's contiguous...)
            isend_prev = isend
            isend += length(to_send)
            send_buf_view = @view send_buf[(isend_prev + 1):isend]
            @assert length(to_send) == length(send_buf_view)
            copyto!(send_buf_view, to_send)
            send_req[n] = MPI.Isend(send_buf_view, rank, tag, comm)
            recv_req[n] = MPI.Irecv!(recv_buf_view, rank, tag, comm)
        end
    end

    @assert isend == length(send_buf)
    @assert irecv == length(recv_buf)

    # 2. Unpack data and perform local transposition.
    # Here we need to know the relative index permutation to go from Pi
    # ordering to Po ordering.
    let perm = relative_permutation(Pi, Po)
        no_perm = is_identity_permutation(perm)

        for m = 1:Nproc
            if m == 1
                n = index_local_req  # copy local data first
            else
                n, status = MPI.Waitany!(recv_req)
            end

            # Non-permuted global indices of received data.
            # TODO avoid repeated code...
            ind = remote_inds[n]
            rrange = intersect.(odims_local, idims[ind])
            rdims = permute_indices(length.(rrange), Pi)

            length_recv_n = prod(rdims)
            off = recv_offsets[n]
            recv_buf_view = @view recv_buf[(off + 1):(off + length_recv_n)]

            # Permuted local indices of output data.
            orange = to_local(Po, rrange)

            src = recv_buf_view
            dest = @view out[orange...]

            # Copy data to `out`, permuting dimensions if required.
            if no_perm
                copyto!(dest, src)
            else
                permutedims!(dest, reshape(src, rdims), perm)
            end
        end
    end

    # Wait for all our data to be sent before returning.
    MPI.Waitall!(send_req)

    out
end
