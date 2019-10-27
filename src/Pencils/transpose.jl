"""
    transpose!(out::PencilArray{T,N}, in::PencilArray{T,N})

Transpose data from one pencil configuration to the other.
"""
transpose!(out::PencilArray{T,N}, in::PencilArray{T,N}) where {T, N} =
    transpose_impl!(out, out.pencil, in, in.pencil)

function assert_compatible(p::Pencil, x::AbstractArray)
    if ndims(x) != 3
        # TODO allow ndims > 3
        throw(ArgumentError("Array must have 3 dimensions."))
    end
    dims = size_local(p)
    if size(x)[1:3] !== dims
        throw(ArgumentError("Array must have dimensions $dims"))
    end
    nothing
end

function assert_compatible(p::Pencil, q::Pencil)
    if p.topology !== q.topology
        throw(ArgumentError("Pencil topologies must be the same."))
    end
    if p.size_global !== q.size_global
        throw(ArgumentError("Global data sizes must be the same between " *
                            "different pencil configurations."))
    end
    nothing
end

"""
    put_colon(::Val{R}, x::NTuple{N}) where {R,N}

Return tuple `x` with the R-th element replaced by a colon.
The return type is known at compile time.

# Example
```jldoctest; setup = :(put_colon = Pencils.put_colon)
julia> put_colon(Val(1), (3, 4, 5))
(Colon(), 4, 5)
julia> put_colon(Val(3), (3, 4, 5))
(3, 4, Colon())
```
"""
function put_colon(::Val{R}, x::NTuple{N}) where {R,N}
    a = ntuple(n -> x[n], Val(R - 1))
    b = ntuple(n -> x[R + n], Val(N - R))
    (a..., :, b...)
end

# TODO generalise to all pencil types
# Transpose Pencil{1} -> Pencil{2}.
# Decomposition dimensions switch from (2, 3) to (1, 3).
function transpose_impl!(out::PencilArray{T,N}, Pout::Pencil{2},
                         in::PencilArray{T,N}, Pin::Pencil{1}) where {T, N}
    @assert in.pencil === Pin
    @assert out.pencil === Pout
    assert_compatible(Pin, Pout)

    # Transposition is performed in the first Cartesian dimension
    # (P1 = 2 -> 1), hence R = 1.
    # TODO what happens if I put Val(2)? Error somewhere?
    @assert Pin.decomp_dims[1] == 2 && Pout.decomp_dims[1] == 1

    # These should be the same (TODO verify!):
    # transpose_impl!(Val(1), out, Pout, in, Pin)
    transpose_impl!(Val(1), out.data, Pout, in.data, Pin)
end

# R: index of MPI subgroup among which the transposition is performed
# Val{R} is put as an argument to make sure that it's a compile-time constant.
function transpose_impl!(::Val{R}, out::AbstractArray{T,N}, Pout::Pencil{2},
                         in::AbstractArray{T,N}, Pin::Pencil{1}) where {R, T, N}
    # Pencil{1} -> Pencil{2} transpose **must** be done via subgroup 1
    @assert R == 1
    @assert N == 3  # for now only this is supported

    @assert Pin.topology === Pout.topology
    topology = Pin.topology
    comm = topology.subcomms[R]  # exchange among the subgroup R
    Nproc = topology.dims[R]
    @assert Nproc == MPI.Comm_size(comm)
    @assert MPI.Comm_rank(comm) < Nproc

    idims_local = Pin.axes_local
    odims_local = Pout.axes_local

    # Example: if coords_local = (2, 3) and R = 1, then ind = (:, 3).
    ind = put_colon(Val(R), topology.coords_local)
    idims = @view Pin.axes_all[ind...]
    odims = @view Pout.axes_all[ind...]

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

    for n in eachindex(send)
        # Global data range that I need to send to process n.
        # Intersections must be done with unpermuted indices!
        # Note: Data is sent and received with the permutation associated to Pin.
        srange = permute_indices((intersect.(idims_local, odims[n])), Pin)

        # Dimensions of data that I will receive from process n.
        rdims = permute_indices(length.(intersect.(odims_local, idims[n])), Pin)

        # TODO avoid copy / allocation!!
        send[n] = in[to_local(Pin, srange)...]
        recv[n] = Array{T,N}(undef, rdims...)

        # Exchange data with the other process (non-blocking operations).
        tag = 42
        rank = topology.subcomm_ranks[R][n]  # actual rank of the other process
        send_req[n] = MPI.Isend(send[n], rank, tag, comm)
        recv_req[n] = MPI.Irecv!(recv[n], rank, tag, comm)
    end

    # TODO
    # - use Waitany! and unpack the data as soon as it's done
    MPI.Waitall!([send_req; recv_req])

    # 2. Unpack data and perform local transposition.
    # Here we need to know the relative index permutation to go from Pin
    # ordering to Pout ordering.
    let perm = relative_permutation(Pin, Pout)
        no_perm = is_identity_permutation(perm)
        for n in eachindex(recv)
            # TODO
            # - avoid repeated operations...
            # - use more consistent variable names with the code above

            # Non-permuted global indices of received data.
            rrange = intersect.(odims_local, idims[n])

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

    out
end
