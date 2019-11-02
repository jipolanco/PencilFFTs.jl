const USE_ALLTOALLV = let key = "PENCILS_USE_ALLTOALLV"
    key in keys(ENV) && ENV[key] == "1"
end

"""
    transpose!(dest::PencilArray{T,N}, src::PencilArray{T,N})

Transpose data from one pencil configuration to the other.

The two pencil configurations must be compatible for transposition:

- they must share the same MPI Cartesian topology,

- they must have the same global data size,

- when written as a sorted tuple, the decomposed dimensions must be almost the
  same, with at most one difference. For instance, if the input of a 3D dataset
  is decomposed in `(2, 3)`, then the output may be decomposed in `(1, 3)`, but
  not in `(1, 2)`. If the decomposed dimensions are the same, then no
  transposition is performed, and data is just copied if needed.

"""
function transpose!(dest::PencilArray{T,N}, src::PencilArray{T,N}) where {T, N}
    dest === src && return dest  # same pencil & same data

    # Verifications
    if src.extra_dims !== dest.extra_dims
        throw(ArgumentError(
            "incompatible number of extra dimensions of PencilArrays: " *
            "$(src.extra_dims) != $(dest.extra_dims)"))
    end
    assert_compatible(src.pencil, dest.pencil)

    # Note: the `decomp_dims` tuples of both pencils must differ by at most one
    # value (as just checked by `assert_compatible`). The transposition is
    # performed along the dimension R where that difference happens.
    R = findfirst(src.pencil.decomp_dims .!= dest.pencil.decomp_dims)

    if R === nothing
        # Both pencil configurations are identical, so we just copy the data,
        # permuting dimensions if needed.
        if get_permutation(src.pencil) === get_permutation(dest.pencil)
            copy!(dest, src)
        else
            perm_base = relative_permutation(src.pencil, dest.pencil)
            perm = prepend_to_permutation(Val(length(src.extra_dims)),
                                          perm_base)
            permutedims!(dest, src, perm)
        end
    else
        # MPI data transposition.
        transpose_impl!(R, dest, src)
    end

    dest
end

function assert_compatible(p::Pencil, q::Pencil)
    if p.topology !== q.topology
        throw(ArgumentError("pencil topologies must be the same."))
    end
    if p.size_global !== q.size_global
        throw(ArgumentError(
            "global data sizes must be the same between different pencil " *
            " configurations. Got $(p.size_global) ≠ $(q.size_global)."))
    end
    # Check that decomp_dims differ on at most one value.
    # Both are expected to be sorted.
    @assert all(issorted.((p.decomp_dims, q.decomp_dims)))
    if sum(p.decomp_dims .!= q.decomp_dims) > 1
        throw(ArgumentError(
            "pencil decompositions must differ in at most one dimension. " *
            "Got decomposed dimensions $(p.decomp_dims) and $(q.decomp_dims)."))
    end
    nothing
end

# Reinterpret UInt8 array as a different type of array.
# This is a workaround to the performance issues when using `reinterpret`.
# See for instance:
# - https://discourse.julialang.org/t/big-overhead-with-the-new-lazy-reshape-reinterpret/7635
# - https://github.com/JuliaLang/julia/issues/28980
unsafe_as_array(::Type{T}, x::Vector{UInt8}, length) where T =
    unsafe_wrap(Array, convert(Ptr{T}, pointer(x)), length, own=false)

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

    @assert in.extra_dims === out.extra_dims
    extra_dims = in.extra_dims
    prod_extra_dims = prod(extra_dims)
    remote_inds = _get_remote_indices(R, topology.coords_local, Nproc)

    idims_local = Pi.axes_local
    odims_local = Po.axes_local

    idims = Pi.axes_all
    odims = Po.axes_all

    # Length of data that I will "send" to myself.
    length_send_local =
        prod(length.(intersect.(idims_local, odims_local))) * prod_extra_dims

    # Total data to be sent / received.
    length_send = length(in) - length_send_local
    length_recv_total = length(out)  # includes local exchange with myself
    length_recv = length_recv_total - length_send_local

    # 1. Send and receive data.
    # Note: I prefer to resize the original UInt8 array instead of the "unsafe"
    # Array{T}.
    resize!(Po.send_buf, sizeof(T) * length_send)
    send_buf = unsafe_as_array(T, Po.send_buf, length_send)
    isend = 0  # current index in send_buf

    resize!(Po.recv_buf, sizeof(T) * length_recv_total)
    recv_buf = unsafe_as_array(T, Po.recv_buf, length_recv_total)
    irecv = 0  # current index in recv_buf
    recv_offsets = Vector{Int}(undef, Nproc)  # all offsets in recv_buf

    index_local_req = -1  # request index associated to local exchange

    if USE_ALLTOALLV
        send_counts = Vector{Cint}(undef, Nproc)
        recv_counts = similar(send_counts)
    else
        send_buf_ptr = pointer(send_buf)
        recv_buf_ptr = pointer(recv_buf)

        # Note: we keep a vector of MPI.MPI_Request (a C object) instead of the
        # MPI.Request wrapper, to avoid allocating a vector each time Waitany
        # and Waitall are called. See MPI_Waitany!.
        send_req = Vector{MPI.MPI_Request}(undef, Nproc)
        recv_req = similar(send_req)
    end

    for (n, ind) in enumerate(remote_inds)
        # Global data range that I need to send to process n.
        srange = intersect.(idims_local, odims[ind])
        length_send_n = prod(length.(srange)) * prod_extra_dims
        local_send_range = to_local(Pi, srange)

        # Determine amount of data to be received.
        rrange = intersect.(odims_local, idims[ind])
        length_recv_n = prod(length.(rrange)) * prod_extra_dims
        recv_offsets[n] = irecv

        rank = topology.subcomm_ranks[R][n]  # actual rank of the other process

        if rank == myrank
            # Copy directly from `in` to `recv_buf`.
            # For convenience, data is put at the end of `recv_buf`. This makes
            # it easier to implement an alternative based on MPI_Alltoallv.
            @assert length_recv_n == length_send_local
            recv_offsets[n] = length_recv
            _copy_to_vec!(recv_buf, length_recv, in, local_send_range,
                          extra_dims)

            if USE_ALLTOALLV
                # Don't send data to myself via Alltoallv.
                send_counts[n] = recv_counts[n] = zero(Cint)
            else
                send_req[n] = recv_req[n] = MPI.REQUEST_NULL.val
                index_local_req = n
            end
        else
            # Copy data into contiguous buffer, then send the buffer.
            # TODO If data inside `in` is contiguous, avoid copying data to buffer,
            # and call MPI.Isend directly. (I need to check if it's contiguous...)
            _copy_to_vec!(send_buf, isend, in, local_send_range, extra_dims)

            irecv += length_recv_n
            isend += length_send_n

            if USE_ALLTOALLV
                send_counts[n] = length_send_n
                recv_counts[n] = length_recv_n
            else
                # Exchange data with the other process (non-blocking operations).
                # Note: data is sent and received with the permutation associated to Pi.
                tag = 42

                # We take the 'val' of the returned MPI.Request, discarding the
                # 'buffer' field.
                send_req[n] =
                    MPI.Isend(send_buf_ptr, length_send_n, rank, tag, comm).val
                recv_req[n] =
                    MPI.Irecv!(recv_buf_ptr, length_recv_n, rank, tag, comm).val

                send_buf_ptr += length_send_n * sizeof(T)
                recv_buf_ptr += length_recv_n * sizeof(T)
            end
        end
    end

    if USE_ALLTOALLV
        # This @view is needed because the Alltoallv wrapper checks that the
        # length of the buffer is consistent with recv_counts.
        recv_buf_view = @view recv_buf[1:length_recv]
        MPI.Alltoallv!(send_buf, recv_buf_view, send_counts, recv_counts, comm)
    else
        @assert isend == length(send_buf) ==
            (send_buf_ptr - pointer(send_buf)) // sizeof(T)
        @assert irecv == length(recv_buf) - length_send_local ==
            (recv_buf_ptr - pointer(recv_buf)) // sizeof(T)
    end

    # 2. Unpack data and perform local transposition.
    # Here we need to know the relative index permutation to go from Pi
    # ordering to Po ordering.
    let perm = relative_permutation(Pi, Po)
        for m = 1:Nproc
            if USE_ALLTOALLV
                n = m
            elseif m == 1
                n = index_local_req  # copy local data first
            else
                n, status = MPI_Waitany!(recv_req)
            end

            # Non-permuted global indices of received data.
            ind = remote_inds[n]
            g_range = intersect.(odims_local, idims[ind])

            length_recv_n = prod(length.(g_range)) * prod_extra_dims
            off = recv_offsets[n]

            # Local output data range in the **input** permutation.
            o_range_iperm =
                permute_indices(to_local(Po, g_range, permute=false), Pi)

            # Copy data to `out`, permuting dimensions if required.
            _copy_from_vec!(out, o_range_iperm, recv_buf, off, perm, extra_dims)
        end
    end

    # Wait for all our data to be sent before returning.
    USE_ALLTOALLV || MPI_Waitall!(send_req)

    out
end

# Cartesian indices of the remote MPI processes included in the subgroup of
# index `R`.
# Example: if coords_local = (2, 3, 5) and R = 1, then this function returns the
# indices corresponding to (:, 3, 5).
function _get_remote_indices(R::Int, coords_local::Dims{M}, Nproc::Int) where M
    t = ntuple(Val(M)) do i
        if i == R
            1:Nproc
        else
            c = coords_local[i]
            c:c
        end
    end
    CartesianIndices(t)
end

function _copy_to_vec!(dest::Vector{T},
                       dest_offset::Int,
                       src::AbstractArray{T,N},
                       src_range::ArrayRegion{P},
                       extra_dims::Dims{E}) where {T,N,P,E}
    @assert P + E == N

    n = dest_offset
    @inbounds for I in CartesianIndices(src_range)
        for K in CartesianIndices(extra_dims)
            dest[n += 1] = src[K, I]
        end
    end

    dest
end

function _copy_from_vec!(dest::AbstractArray{T,N},
                         o_range_iperm::ArrayRegion{P},
                         src::Vector{T},
                         src_offset::Int,
                         perm::OptionalPermutation{P},
                         extra_dims::Dims{E}) where {T,N,P,E}
    @assert P + E == N

    # The idea is to visit `dest` not in its natural order (with the fastest
    # dimension first), but with a permutation corresponding to the layout of
    # the `src` data.
    n = src_offset
    @inbounds for I in CartesianIndices(o_range_iperm)
        # Switch from input to output permutation.
        # Note: this should have zero cost if perm == nothing.
        J = permute_indices(I, perm)
        for K in CartesianIndices(extra_dims)
            dest[K, J] = src[n += 1]
        end
    end

    dest
end
