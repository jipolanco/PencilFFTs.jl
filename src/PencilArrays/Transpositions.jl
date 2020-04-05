module Transpositions

import LinearAlgebra: transpose!

using TimerOutputs
import MPI

using ..PencilArrays
import ..PencilArrays:
    ArrayRegion,
    relative_permutation,
    permute_indices,
    same_permutation,
    append_to_permutation,
    extract

# Declare transposition approaches.
abstract type AbstractTransposeMethod end

struct IsendIrecv <: AbstractTransposeMethod end
struct Alltoallv <: AbstractTransposeMethod end

function Base.show(io::IO, ::T) where {T<:AbstractTransposeMethod}
    print(io, nameof(T))
end

"""
    transpose!(dest::PencilArray{T,N}, src::PencilArray{T,N};
               method=Transpositions.IsendIrecv())

Transpose data from one pencil configuration to the other.

The two pencil configurations must be compatible for transposition:

- they must share the same MPI Cartesian topology,

- they must have the same global data size,

- when written as a sorted tuple, the decomposed dimensions must be almost the
  same, with at most one difference. For instance, if the input of a 3D dataset
  is decomposed in `(2, 3)`, then the output may be decomposed in `(1, 3)`, but
  not in `(1, 2)`. If the decomposed dimensions are the same, then no
  transposition is performed, and data is just copied if needed.

The `src` and `dest` arrays may be aliased (they can share memory space).

# Performance tuning

The `method` argument allows to choose between transposition implementations.
This can be useful to tune performance of MPI data transfers.
Two values are currently accepted:

- `Transpositions.IsendIrecv()` uses non-blocking point-to-point data transfers
  (`MPI_Isend` and `MPI_Irecv`).
  This may be more performant since data transfers are interleaved with local
  data transpositions (index permutation of received data).
  This is the default.

- `Transpositions.Alltoallv()` uses `MPI_Alltoallv` for global data
  transpositions.

"""
function transpose!(
        dest::PencilArray{T,N}, src::PencilArray{T,N};
        method::AbstractTransposeMethod=IsendIrecv(),
    ) where {T,N}
    dest === src && return dest  # same pencil & same data

    Pi = pencil(src)
    Po = pencil(dest)
    timer = get_timer(Pi)

    @timeit_debug timer "transpose!" begin
        # Verifications
        if src.extra_dims !== dest.extra_dims
            throw(ArgumentError(
                "incompatible number of extra dimensions of PencilArrays: " *
                "$(src.extra_dims) != $(dest.extra_dims)"))
        end

        assert_compatible(Pi, Po)

        # Note: the `decomp_dims` tuples of both pencils must differ by at most
        # one value (as just checked by `assert_compatible`). The transposition
        # is performed along the dimension R where that difference happens.
        R = findfirst(Pi.decomp_dims .!= Po.decomp_dims)
        transpose_impl!(R, dest, src, method=method)
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
# The input array should have enough space for the reinterpreted array with the
# given dimensions.
# This is a workaround to the performance issues when using `reinterpret`.
# See for instance:
# - https://discourse.julialang.org/t/big-overhead-with-the-new-lazy-reshape-reinterpret/7635
# - https://github.com/JuliaLang/julia/issues/28980
function unsafe_as_array(::Type{T}, x::Vector{UInt8}, dims) where T
    p = Ptr{T}(pointer(x))
    A = unsafe_wrap(Array, p, dims, own=false)
    @assert sizeof(A) <= sizeof(x)
    A
end

# Only local transposition.
function transpose_impl!(::Nothing, out::PencilArray{T,N}, in::PencilArray{T,N};
                         kwargs...) where {T,N}
    Pi = pencil(in)
    Po = pencil(out)
    timer = get_timer(Pi)

    # Both pencil configurations are identical, so we just copy the data,
    # permuting dimensions if needed.
    @assert size(in) === size(out)
    ui = parent(in)
    uo = parent(out)

    if same_permutation(get_permutation(Pi), get_permutation(Po))
        @timeit_debug timer "copy!" copy!(uo, ui)
    else
        @timeit_debug timer "permute_local!" permute_local!(out, in)
    end

    out
end

function permute_local!(out::PencilArray{T,N},
                        in::PencilArray{T,N}) where {T, N}
    Pi = pencil(in)
    Po = pencil(out)

    perm = let perm_base = relative_permutation(Pi, Po)
        p = append_to_permutation(perm_base, Val(length(in.extra_dims)))
        extract(p) :: Tuple
    end

    ui = parent(in)
    uo = parent(out)

    inplace = Base.mightalias(ui, uo)

    if inplace
        # TODO optimise in-place version?
        # For now we permute into a temporary buffer, and then we copy to `out`.
        # We reuse `recv_buf` used for MPI transposes.
        buf = let x = Pi.recv_buf
            n = length(uo)
            dims = size(uo)
            resize!(x, sizeof(T) * n)
            vec = unsafe_as_array(T, x, n)
            reshape(vec, dims)
        end
        permutedims!(buf, ui, perm)
        copy!(uo, buf)
    else
        # Permute directly onto the output.
        permutedims!(uo, ui, perm)
    end

    out
end

_mpi_buffer(p::Ptr{T}, count) where {T} =
    MPI.Buffer(p, Cint(count), MPI.Datatype(T))

# Transposition among MPI processes in a subcommunicator.
# R: index of MPI subgroup (dimension of MPI Cartesian topology) along which the
# transposition is performed.
function transpose_impl!(
        R::Int, Ao::PencilArray{T,N}, Ai::PencilArray{T,N};
        method::AbstractTransposeMethod=IsendIrecv(),
    ) where {T,N}
    Pi = pencil(Ai)
    Po = pencil(Ao)
    timer = get_timer(Pi)

    @assert Pi.topology === Po.topology
    @assert Ai.extra_dims === Ao.extra_dims

    topology = Pi.topology
    comm = topology.subcomms[R]  # exchange among the subgroup R
    Nproc = topology.dims[R]
    subcomm_ranks = topology.subcomm_ranks[R]
    myrank = subcomm_ranks[topology.coords_local[R]]  # rank in subgroup

    remote_inds = _get_remote_indices(R, topology.coords_local, Nproc)

    # Length of data that I will "send" to myself.
    length_self = let range_intersect = intersect.(Pi.axes_local, Po.axes_local)
        prod(length.(range_intersect)) * prod(extra_dims(Ai))
    end

    # Total data to be sent / received.
    length_send = length(Ai) - length_self
    length_recv_total = length(Ao)  # includes local exchange with myself

    resize!(Po.send_buf, sizeof(T) * length_send)
    send_buf = unsafe_as_array(T, Po.send_buf, length_send)

    resize!(Po.recv_buf, sizeof(T) * length_recv_total)
    recv_buf = unsafe_as_array(T, Po.recv_buf, length_recv_total)
    recv_offsets = Vector{Int}(undef, Nproc)  # all offsets in recv_buf

    req_length = method === Alltoallv() ? 0 : Nproc
    send_req = Vector{MPI.Request}(undef, req_length)
    recv_req = similar(send_req)

    buffers = (send_buf, recv_buf)
    requests = (send_req, recv_req)

    # 1. Pack and send data.
    @timeit_debug timer "pack data" index_local_req = _transpose_send!(
        buffers, recv_offsets, requests, length_self, remote_inds,
        (comm, subcomm_ranks, myrank),
        Ao, Ai, method, timer,
    )

    # 2. Unpack data and perform local transposition.
    @timeit_debug timer "unpack data" _transpose_recv!(
        recv_buf, recv_offsets, recv_req,
        remote_inds, index_local_req,
        Ao, Ai, method, timer,
    )

    # Wait for all our data to be sent before returning.
    if !isempty(send_req)
        @timeit_debug timer "wait send" MPI.Waitall!(send_req)
    end

    Ao
end

function _transpose_send!(
        (send_buf, recv_buf),
        recv_offsets, requests,
        length_self, remote_inds,
        (comm, subcomm_ranks, myrank),
        Ao::PencilArray{T}, Ai::PencilArray{T},
        method::AbstractTransposeMethod,
        timer::TimerOutput,
    ) where {T}
    Pi = pencil(Ai)  # input (sent data)
    Po = pencil(Ao)  # output (received data)

    idims_local = Pi.axes_local
    odims_local = Po.axes_local

    idims = Pi.axes_all
    odims = Po.axes_all

    exdims = extra_dims(Ai)
    prod_extra_dims = prod(exdims)

    isend = 0  # current index in send_buf
    irecv = 0  # current index in recv_buf

    index_local_req = -1  # request index associated to local exchange

    # Data received from other processes.
    length_recv = length(Ao) - length_self

    Nproc = length(subcomm_ranks)
    @assert Nproc == MPI.Comm_size(comm)
    @assert myrank == MPI.Comm_rank(comm)

    buf_info = _make_buffer_info(method, (send_buf, recv_buf), Nproc)

    for (n, ind) in enumerate(remote_inds)
        # Global data range that I need to send to process n.
        srange = intersect.(idims_local, odims[ind])
        length_send_n = prod(length.(srange)) * prod_extra_dims
        local_send_range = to_local(Pi, srange, permute=true)

        # Determine amount of data to be received.
        rrange = intersect.(odims_local, idims[ind])
        length_recv_n = prod(length.(rrange)) * prod_extra_dims
        recv_offsets[n] = irecv

        rank = subcomm_ranks[n]  # actual rank of the other process

        if rank == myrank
            # Copy directly from `Ai` to `recv_buf`.
            # For convenience, data is put at the end of `recv_buf`.
            # This makes it easier to implement an alternative based on MPI_Alltoallv.
            @assert length_recv_n == length_self
            recv_offsets[n] = length_recv
            @timeit_debug timer "copy_range!" copy_range!(
                recv_buf, length_recv, Ai, local_send_range, exdims, timer)
            _transpose_send_self!(method, n, requests, buf_info)
            index_local_req = n
        else
            # Copy data into contiguous buffer, then send the buffer.
            @timeit_debug timer "copy_range!" copy_range!(
                send_buf, isend, Ai, local_send_range, exdims, timer)
            _transpose_send_other!(
                method, buf_info, (length_send_n, length_recv_n), n,
                requests, (rank, comm), eltype(Ai),
            )
            irecv += length_recv_n
            isend += length_send_n
        end
    end

    if method === Alltoallv()
        # This @view is needed because the Alltoallv wrapper checks that the
        # length of the buffer is consistent with recv_counts.
        recv_buf_view = @view recv_buf[1:length_recv]
        @timeit_debug timer "MPI.Alltoallv!" MPI.Alltoallv!(
            send_buf, recv_buf_view,
            buf_info.send_counts, buf_info.recv_counts, comm,
        )
    end

    index_local_req
end

function _make_buffer_info(::IsendIrecv, (send_buf, recv_buf), Nproc)
    (
        send_ptr = Ref(pointer(send_buf)),
        recv_ptr = Ref(pointer(recv_buf)),
    )
end

function _make_buffer_info(::Alltoallv, bufs, Nproc)
    counts = Vector{Cint}(undef, Nproc)
    (
        send_counts = counts,
        recv_counts = similar(counts),
    )
end

function _transpose_send_self!(::IsendIrecv, n, (send_req, recv_req), etc...)
    send_req[n] = recv_req[n] = MPI.REQUEST_NULL
    nothing
end

function _transpose_send_self!(::Alltoallv, n, reqs, buf_info)
    # Don't send data to myself via Alltoallv.
    buf_info.send_counts[n] = buf_info.recv_counts[n] = zero(Cint)
    nothing
end

function _transpose_send_other!(
        ::IsendIrecv, buf_info, (length_send_n, length_recv_n),
        n, (send_req, recv_req), (rank, comm), ::Type{T}
    ) where {T}
    # Exchange data with the other process (non-blocking operations).
    # Note: data is sent and received with the permutation associated to Pi.
    tag = 42
    send_req[n] = MPI.Isend(
        _mpi_buffer(buf_info.send_ptr[], length_send_n),
        rank, tag, comm
    )
    recv_req[n] = MPI.Irecv!(
        _mpi_buffer(buf_info.recv_ptr[], length_recv_n),
        rank, tag, comm
    )
    buf_info.send_ptr[] += length_send_n * sizeof(T)
    buf_info.recv_ptr[] += length_recv_n * sizeof(T)
    nothing
end

function _transpose_send_other!(
        ::Alltoallv, buf_info, (length_send_n, length_recv_n), n, args...
    )
    buf_info.send_counts[n] = length_send_n
    buf_info.recv_counts[n] = length_recv_n
    nothing
end

function _transpose_recv!(
        recv_buf, recv_offsets, recv_req,
        remote_inds, index_local_req,
        Ao::PencilArray, Ai::PencilArray,
        method::AbstractTransposeMethod,
        timer::TimerOutput,
    )
    Pi = pencil(Ai)  # input (sent data)
    Po = pencil(Ao)  # output (received data)

    odims_local = Po.axes_local
    idims = Pi.axes_all

    exdims = extra_dims(Ao)
    prod_extra_dims = prod(exdims)

    # Relative index permutation to go from Pi ordering to Po ordering.
    perm = relative_permutation(Pi, Po)

    Nproc = length(remote_inds)

    for m = 1:Nproc
        if method === Alltoallv()
            n = m
        elseif m == 1
            n = index_local_req  # copy local data first
        else
            @timeit_debug timer "wait receive" n, status =
                MPI.Waitany!(recv_req)
        end

        # Non-permuted global indices of received data.
        ind = remote_inds[n]
        g_range = intersect.(odims_local, idims[ind])

        length_recv_n = prod(length.(g_range)) * prod_extra_dims
        off = recv_offsets[n]

        # Local output data range in the **input** permutation.
        o_range_iperm =
            permute_indices(to_local(Po, g_range, permute=false), Pi)

        # Copy data to `Ao`, permuting dimensions if required.
        @timeit_debug timer "copy_permuted!" copy_permuted!(
            Ao, o_range_iperm, recv_buf, off, perm, exdims, timer)
    end

    Ao
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

function copy_range!(dest::Vector{T}, dest_offset::Int, src::PencilArray{T,N},
                     src_range::ArrayRegion{P}, extra_dims::Dims{E}, timer,
                    ) where {T,N,P,E}
    @assert P + E == N

    n = dest_offset
    src_p = parent(src)  # array with non-permuted indices
    for K in CartesianIndices(extra_dims)
        for I in CartesianIndices(src_range)
            @inbounds dest[n += 1] = src_p[I, K]
        end
    end

    dest
end

function copy_permuted!(dest::PencilArray{T,N}, o_range_iperm::ArrayRegion{P},
                        src::Vector{T}, src_offset::Int,
                        perm::Union{Nothing, Val}, extra_dims::Dims{E},
                        timer) where {T,N,P,E}
    @assert P + E == N

    # The idea is to visit `dest` not in its natural order (with the fastest
    # dimension first), but with a permutation corresponding to the layout of
    # the `src` data.
    n = src_offset
    dest_p = parent(dest)  # array with non-permuted indices
    for K in CartesianIndices(extra_dims)
        for I in CartesianIndices(o_range_iperm)
            # Switch from input to output permutation.
            # Note: this should have zero cost if perm == nothing.
            J = permute_indices(I, perm)
            @inbounds dest_p[J, K] = src[n += 1]
        end
    end

    dest
end

end  # module Transpositions

# Deprecated since v0.5.
Base.@deprecate_binding TransposeMethods Transpositions
