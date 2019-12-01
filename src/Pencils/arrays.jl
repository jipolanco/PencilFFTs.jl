"""
    PencilArray(pencil::Pencil, data::AbstractArray{T,N})

Create array wrapper with pencil decomposition information.

The array dimensions and element type must be consistent with those of the given
pencil.

!!! note "Index permutations"

    If the `Pencil` has an associated index permutation, then the input array
    must have its dimensions permuted accordingly. In any case, the resulting
    `PencilArray` is accessed with non-permuted indices.

    Note that the original array can be recovered using [`parent`](@ref), and
    must be accessed using permuted indices.

    ##### Example

    Suppose `pencil` has local dimensions `(10, 20, 30)` before permutation, and
    has an asociated permutation `(2, 3, 1)`.
    Then:
    ```julia
    data = zeros(20, 30, 10)       # parent array (with permuted dimensions)
    u = PencilArray(pencil, data)  # wrapper with dimensions (10, 20, 30)
    u[15, 25, 5]                   # BoundsError (15 > 10 and 25 > 20)
    u[5, 15, 25]                   # correct
    parent(u)[15, 25, 5]           # correct
    ```

!!! note "Extra dimensions"

    The data array can have one or more extra dimensions to the right (slow
    indices).
    For instance, these may correspond to vector or tensor components.
    These dimensions are not affected by permutations.

    ##### Example

    ```julia
    dims = (20, 30, 10)
    PencilArray(pencil, zeros(dims...))        # works (scalar)
    PencilArray(pencil, zeros(dims..., 3))     # works (3-component vector)
    PencilArray(pencil, zeros(dims..., 4, 3))  # works (4×3 tensor)
    PencilArray(pencil, zeros(3, dims...))     # fails
    ```

---

    PencilArray(pencil::Pencil, [extra_dims=()])

Allocate uninitialised `PencilArray` that can hold data in the local pencil.

Extra dimensions, for instance representing vector components, can be specified.
These dimensions are added to the rightmost (slowest) indices of the resulting
array.

# Example
Suppose `pencil` has local dimensions `(10, 20, 30)` after permutation. Then:
```julia
PencilArray(pencil)          # array dimensions are (10, 20, 30)
PencilArray(pencil, (4, 3))  # array dimensions are (10, 20, 30, 4, 3)
```
"""
struct PencilArray{T, N,
                   E,   # number of "extra" dimensions (= N - Np)
                   Np,  # number of "spatial" dimensions (i.e. dimensions of the Pencil)
                   P <: Pencil,
                   A <: AbstractArray{T,N},
                   OP <: OptionalPermutation{Np},
                  } <: AbstractArray{T,N}
    pencil   :: P
    perm     :: OP
    perm_inv :: OP
    data     :: A
    extra_dims :: Dims{E}

    function PencilArray(pencil::Pencil{Np, Mp, T} where {Np, Mp},
                         data::AbstractArray{T, N}) where {T, N}
        P = typeof(pencil)
        A = typeof(data)
        Np = ndims(pencil)
        E = N - Np
        size_data = size(data)

        geom_dims = ntuple(n -> size_data[n], Np)  # = size_data[1:Np]
        extra_dims = ntuple(n -> size_data[Np + n], E)  # = size_data[Np+1:N]

        if geom_dims !== size_local(pencil)
            throw(DimensionMismatch(
                "array has incorrect dimensions: $(size_data). " *
                "Local dimensions of pencil: $(size_local(pencil))."))
        end

        perm = get_permutation(pencil)
        perm_inv = inverse_permutation(perm)
        OP = typeof(perm)

        new{T, N, E, Np, P, A, OP}(pencil, perm, perm_inv, data, extra_dims)
    end
end

PencilArray(pencil::Pencil, extra_dims::Dims=()) =
    PencilArray(pencil, Array{eltype(pencil)}(undef, size_local(pencil)...,
                                              extra_dims...))

function Base.size(x::PencilArray)
    s = _permute_indices(x, size(parent(x))...; perm=x.perm_inv)
    @assert s === (size_local(x.pencil, permute=false)..., x.extra_dims...)
    s
end

# Use same index style as the parent array.
Base.IndexStyle(::Type{<:PencilArray{T,N,E,Np,P,A}}
                where {T,N,E,Np,P}) where {A} = IndexStyle(A)

@propagate_inbounds @inline Base.getindex(x::PencilArray, inds...) =
    x.data[_make_indices(x, inds)...]
@propagate_inbounds @inline Base.setindex!(x::PencilArray, v, inds...) =
    x.data[_make_indices(x, inds)...] = v

@inline _make_indices(x::PencilArray, I::Tuple) =
    _permute_indices(x, _splat_indices(I...)...)

@inline _make_indices(x::PencilArray, I) =
    throw(ArgumentError("unsupported index: $I"))

# Linear indexing
@inline _make_indices(::PencilArray,
                      i::Tuple{Union{Integer, AbstractArray}}) =
    (Base.to_index(i...), )

# Permute "permutable" indices of array, i.e., excluding the last E dimensions.
function _permute_indices(
        x::PencilArray{T,N,E,Np} where T,
        I::Vararg{Any,N};
        perm=x.perm,
       ) where {N,E,Np}
    a = ntuple(d -> I[d + Np], Val(E))
    b = ntuple(d -> I[d], Val(Np))
    c = permute_indices(b, perm)
    (c..., a...)
end

@inline _splat_indices(i::CartesianIndex, inds...) =
    (Tuple(i)..., _splat_indices(inds...)...)
@inline _splat_indices(i, inds...) = (i, _splat_indices(inds...)...)
@inline _splat_indices() = ()

function Base.similar(x::PencilArray{T,N} where T, ::Type{S},
                      dims_non_permuted::Dims{N}) where {S,N}
    dims_parent = _permute_indices(x, dims_non_permuted...;
                                   perm=x.perm)
    @assert dims_non_permuted !== size(x) || dims_parent === size(parent(x))
    PencilArray(x.pencil, similar(x.data, S, dims_parent))
end

"""
    pencil(x::PencilArray)

Returns decomposition configuration associated to the `PencilArray`.
"""
pencil(x::PencilArray) = x.pencil

"""
    parent(x::PencilArray)

Returns array wrapped by a `PencilArray`.

!!! note "Index permutations"

    If the underlying pencil configuration has an associated permutation, then
    the parent array must be accessed with permuted indices.

    See [`PencilArray`](@ref) for more details.
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
size_global(x::PencilArray) = (size_global(x.pencil)..., x.extra_dims...)

"""
    range_local(x::PencilArray; permute=true)

Local data range held by the PencilArray.

By default the dimensions are permuted to match the order of indices in the
array.
"""
range_local(x::PencilArray; permute=true) =
    (range_local(pencil(x), permute=permute)..., Base.OneTo.(x.extra_dims)...)

"""
    get_comm(x::PencilArray)

Get MPI communicator associated to a pencil-distributed array.
"""
get_comm(x::PencilArray) = get_comm(x.pencil)

"""
    spatial_indices(x::PencilArray)
    spatial_indices(x::GlobalPencilArray)

Create a `CartesianIndices` to iterate over the local "spatial" dimensions of a
pencil-decomposed array.

The "spatial" dimensions are those that may be decomposed (as opposed to the
"extra" dimensions, which are not considered by this function).
"""
spatial_indices(x::PencilArray{T,N,E,Np} where {T,N,E}) where {Np} =
    CartesianIndices(ntuple(n -> axes(x, n), Val(Np)))

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
            p = append_to_permutation(invperm, Val(length(extra_dims)))
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
            recv[n] = Array{T,N}(undef, rdims..., extra_dims...)
            recv_req[n] = MPI.Irecv!(recv[n], src_rank, mpi_tag, comm)
        end
    end

    # Unpack data.
    dest = Array{T,N}(undef, size_global(x))

    # Copy local data.
    colons_extra_dims = ntuple(n -> Colon(), Val(length(extra_dims)))
    dest[pen.axes_local..., colons_extra_dims...] .= data

    # Copy remote data.
    for m = 2:Nproc
        n, status = MPI.Waitany!(recv_req)
        rrange = pen.axes_all[n]
        dest[rrange..., colons_extra_dims...] .= recv[n]
    end

    end  # @timeit_debug

    dest
end
