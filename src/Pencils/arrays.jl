"""
    PencilArray(pencil::Pencil, data::AbstractArray{T,N})

Create array wrapper with pencil decomposition information.

The array dimensions and element type must be consistent with those of the given
pencil.

!!! note "Index permutations"

    If the `Pencil` has an associated index permutation, then the input array
    must have its dimensions permuted accordingly.
    The resulting `PencilArray` must be accessed with permuted indices, just
    like its parent array.

    ##### Example

    Suppose `pencil` has local dimensions `(10, 20, 30)` before permutation, and
    has an asociated permutation `(2, 3, 1)`.
    Then:
    ```julia
    data = zeros(20, 30, 10)       # parent array (with permuted dimensions)
    u = PencilArray(pencil, data)  # wrapper with dimensions (10, 20, 30)
    u[5, 15, 25]                   # BoundsError (25 > 10)
    u[15, 25, 5]                   # correct
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
Suppose `pencil` has local dimensions `(20, 10, 30)` after permutation. Then:
```julia
PencilArray(pencil)          # array dimensions are (20, 10, 30)
PencilArray(pencil, (4, 3))  # array dimensions are (20, 10, 30, 4, 3)
```
"""
struct PencilArray{T, N,
                   P <: Pencil,
                   A <: AbstractArray{T,N},
                   E,   # number of "extra" dimensions (= N - Np)
                   Np,  # number of "spatial" dimensions (i.e. dimensions of the Pencil)
                  } <: AbstractArray{T,N}
    pencil   :: P
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

        new{T, N, P, A, E, Np}(pencil, data, extra_dims)
    end
end

PencilArray(pencil::Pencil, extra_dims::Dims=()) =
    PencilArray(pencil, Array{eltype(pencil)}(undef, size_local(pencil)...,
                                              extra_dims...))

"""
    size(x::PencilArray)

Return local dimensions of a `PencilArray`.
"""
Base.size(x::PencilArray) = size(parent(x))

# Use same index style as the parent array.
Base.IndexStyle(::Type{<:PencilArray{T,N,P,A}} where {T,N,P}) where {A} =
    IndexStyle(A)

@propagate_inbounds Base.getindex(x::PencilArray, inds...) = x.data[inds...]
@propagate_inbounds Base.setindex!(x::PencilArray, v, inds...) =
    x.data[inds...] = v

Base.similar(x::PencilArray, ::Type{S}, dims::Dims) where S =
    PencilArray(x.pencil, similar(x.data, S, dims))

"""
    pencil(x::PencilArray)

Return decomposition configuration associated to the `PencilArray`.
"""
pencil(x::PencilArray) = x.pencil

"""
    parent(x::PencilArray)

Return array wrapped by a `PencilArray`.
"""
Base.parent(x::PencilArray) = x.data

"""
    ndims_extra(x::PencilArray)

Number of "extra" dimensions associated to `PencilArray`.

These are the dimensions that are not associated to the domain geometry.
For instance, they may correspond to vector or tensor components.

These dimensions correspond to the rightmost indices of the array.

The total number of dimensions of a `PencilArray` is given by:

    ndims(x) == ndims_space(x) + ndims_extra(x)

"""
ndims_extra(x::PencilArray) = length(x.extra_dims)

"""
    ndims_space(x::PencilArray)

Number of dimensions associated to the domain geometry.

These dimensions correspond to the leftmost indices of the array.

The total number of dimensions of a `PencilArray` is given by:

    ndims(x) == ndims_space(x) + ndims_extra(x)

"""
ndims_space(x::PencilArray) = ndims(x) - ndims_extra(x)

"""
    extra_dims(x::PencilArray)

Return tuple with size of "extra" dimensions of `PencilArray`.
"""
extra_dims(x::PencilArray) = x.extra_dims

"""
    size_global(x::PencilArray; permute=false)

Global dimensions associated to the given array.

Unlike `size`, by default the returned dimensions are *not* permuted according
to the associated pencil configuration.
"""
size_global(x::PencilArray; permute=false) =
    (size_global(x.pencil, permute=permute)..., x.extra_dims...)

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
    get_permutation(x::PencilArray)

Get index permutation associated to the given `PencilArray`.

Returns `nothing` if there is no associated permutation.
"""
get_permutation(x::PencilArray) = get_permutation(pencil(x))

"""
    spatial_indices(x::PencilArray)
    spatial_indices(x::GlobalPencilArray)

Create a `CartesianIndices` to iterate over the local "spatial" dimensions of a
pencil-decomposed array.

The "spatial" dimensions are those that may be decomposed (as opposed to the
"extra" dimensions, which are not considered by this function).
"""
function spatial_indices(x::PencilArray)
    Np = ndims_space(x)
    CartesianIndices(ntuple(n -> axes(x, n), Val(Np)))
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
            p = append_to_permutation(invperm, Val(length(extra_dims)))
            permutedims(x.data, extract(p))  # creates copy!
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
