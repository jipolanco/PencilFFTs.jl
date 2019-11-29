"""
    PencilArray(pencil::P, data::AbstractArray{T,N})

Create array wrapper with pencil decomposition information.

The array dimensions and element type must be consistent with those of the given
pencil.

The data array can have one or more extra dimensions to the left (fast indices).
For instance, these may correspond to vector or tensor components.

# Example

Suppose `pencil` has local dimensions `(10, 20, 30)`. Then:
```julia
PencilArray(pencil, zeros(10, 20, 30))        # works (scalar)
PencilArray(pencil, zeros(3, 10, 20, 30))     # works (3-component vector)
PencilArray(pencil, zeros(4, 3, 10, 20, 30))  # works (4×3 tensor)
PencilArray(pencil, zeros(10, 20, 30, 3))     # fails
```

---

    PencilArray(pencil::Pencil, [extra_dims...])

Allocate uninitialised `PencilArray` that can hold data in the local pencil.

Extra dimensions, for instance representing vector components, can be specified.
These dimensions are added to the leftmost (fastest) indices of the resulting
array.

# Example
Suppose `pencil` has local dimensions `(10, 20, 30)`. Then:
```julia
PencilArray(pencil)        # array dimensions are (10, 20, 30)
PencilArray(pencil, 4, 3)  # array dimensions are (4, 3, 10, 20, 30)
```
"""
struct PencilArray{T, N,
                   P <: Pencil,
                   A <: AbstractArray{T,N},
                   E,  # number of "extra" dimensions (>= 0)
                  } <: AbstractArray{T,N}
    pencil :: P
    data   :: A
    extra_dims :: Dims{E}

    function PencilArray(pencil::Pencil{Np, Mp, T} where {Np, Mp},
                         data::AbstractArray{T, N}) where {T, N}
        P = typeof(pencil)
        A = typeof(data)
        ndims_space = ndims(pencil)
        E = N - ndims_space
        size_data = size(data)

        extra_dims = ntuple(n -> size_data[n], E)  # = size_data[1:E]
        geom_dims = ntuple(n -> size_data[E + n], ndims_space)  # = size_data[E+1:end]

        if geom_dims !== size_local(pencil)
            throw(DimensionMismatch(
                "array has incorrect dimensions: $(size_data). " *
                "Local dimensions of pencil: $(size_local(pencil))."))
        end

        new{T, N, P, A, E}(pencil, data, extra_dims)
    end
end

PencilArray(pencil::Pencil, extra_dims::Vararg{Int}) =
    PencilArray(pencil, Array{eltype(pencil)}(undef, extra_dims...,
                                              size_local(pencil)...))

Base.size(x::PencilArray) = size(parent(x))

# Use same index style as the parent array.
Base.IndexStyle(::Type{<:PencilArray{T,N,P,A}} where {T,N,P}) where A =
    IndexStyle(A)

@propagate_inbounds Base.getindex(x::PencilArray, inds...) = x.data[inds...]
@propagate_inbounds Base.setindex!(x::PencilArray, v, inds...) =
    x.data[inds...] = v

Base.similar(x::PencilArray, ::Type{S}, dims::Dims) where S =
    PencilArray(x.pencil, similar(x.data, S, dims))

"""
    pencil(x::PencilArray)

Returns decomposition configuration associated to the `PencilArray`.
"""
pencil(x::PencilArray) = x.pencil

"""
    parent(x::PencilArray)

Returns array wrapped by a `PencilArray`.
"""
Base.parent(x::PencilArray) = x.data

"""
    ndims_extra(x::PencilArray)

Number of "extra" dimensions associated to `PencilArray`.

These are the dimensions that are not associated to the domain geometry.
For instance, they may correspond to vector or tensor components.
"""
ndims_extra(x::PencilArray) = length(x.extra_dims)

"""
    size_global(x::PencilArray)

Global dimensions associated to the given array.

Unlike `size`, the returned dimensions are *not* permuted according to the
associated pencil configuration.
"""
size_global(x::PencilArray) = (x.extra_dims..., size_global(x.pencil)...)

"""
    range_local(x::PencilArray; permute=true)

Local data range held by the PencilArray.

By default the dimensions are permuted to match the order of indices in the
array.
"""
range_local(x::PencilArray; permute=true) =
    (Base.OneTo.(x.extra_dims)..., range_local(pencil(x), permute=permute)...)

"""
    get_comm(x::PencilArray)

Get MPI communicator associated to a pencil-distributed array.
"""
get_comm(x::PencilArray) = get_comm(x.pencil)

"""
    global_view(x::PencilArray)

Create an [`OffsetArray`](https://github.com/JuliaArrays/OffsetArrays.jl) of a
`PencilArray` that takes global indices.

The order of indices in the returned view is the same as for the original array
`x`. That is, if the indices of `x` are permuted, so are those of the returned
array.
"""
function global_view(x::PencilArray)
    r = range_local(x)
    offsets = first.(r) .- 1
    xo = OffsetArray(x, offsets)
    @assert parent(xo) === x  # OffsetArray shouldn't create a copy...
    xo
end

"""
    gather(x::PencilArray, [root::Integer=0])

Gather data from all MPI processes into one (big) array.

Data is received by the `root` process.

Returns the full array on the `root` process, and `nothing` on the other
processes.

This can be useful for testing, but it shouldn't be used with very large
datasets!
"""
function gather(x::PencilArray{T,N}, root::Integer=0) where {T, N}

    timer = get_timer(pencil(x))

    @timeit_debug timer "gather" begin

    # TODO reduce allocations! see `transpose_impl!`
    comm = get_comm(x)
    rank = MPI.Comm_rank(comm)
    mpi_tag = 42
    pen = pencil(x)
    extra_dims = x.extra_dims

    # Each process sends its data to the root process.
    # If the local indices are permuted, the permutation is reverted before
    # sending the data.
    data = let perm = pen.perm
        if is_identity_permutation(perm)
            x.data
        else
            # Apply inverse permutation.
            invperm = relative_permutation(perm, nothing)
            p = prepend_to_permutation(Val(length(extra_dims)), invperm)
            permutedims(x.data, p)  # creates copy!
        end
    end

    if rank != root
        # Wait for data to be sent, then return.
        send_req = MPI.Isend(data, root, mpi_tag, comm)
        MPI.Wait!(send_req)
        return nothing
    end

    # Receive data (root only).
    topo = pen.topology
    Nproc = length(topo)
    recv = Vector{Array{T,N}}(undef, Nproc)
    recv_req = Vector{MPI.Request}(undef, Nproc)

    root_index = -1

    for n = 1:Nproc
        # Global data range that I will receive from process n.
        rrange = pen.axes_all[n]
        rdims = length.(rrange)

        src_rank = topo.ranks[n]  # actual rank of sending process
        if src_rank == root
            root_index = n
            recv_req[n] = MPI.REQUEST_NULL
        else
            # TODO avoid allocation?
            recv[n] = Array{T,N}(undef, extra_dims..., rdims...)
            recv_req[n] = MPI.Irecv!(recv[n], src_rank, mpi_tag, comm)
        end
    end

    # Unpack data.
    dest = Array{T,N}(undef, size_global(x))

    # Copy local data.
    colons_extra_dims = ntuple(n -> Colon(), Val(length(extra_dims)))
    dest[colons_extra_dims..., pen.axes_local...] .= data

    # Copy remote data.
    for m = 2:Nproc
        n, status = MPI.Waitany!(recv_req)
        rrange = pen.axes_all[n]
        dest[colons_extra_dims..., rrange...] .= recv[n]
    end

    end  # @timeit_debug

    dest
end
