# Functions implemented for PencilArray.
import Base: size, getindex, setindex!, similar, IndexStyle

"""
    PencilArray(pencil::P, data::AbstractArray{T,N})

Create array wrapper with pencil decomposition information.

The array dimensions must be consistent with the dimensions of the local pencil.

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

    PencilArray(pencil::Pencil, [T=Float64], [extra_dims...])

Allocate uninitialised PencilArray that can hold data in the local pencil.

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
struct PencilArray{T, N, A<:AbstractArray{T,N},
                   P<:Pencil,
                   E,  # number of "extra" dimensions (>= 0)
                  } <: AbstractArray{T,N}
    pencil :: P
    data   :: A
    extra_dims :: Dims{E}

    function PencilArray(pencil::P,
                         data::AbstractArray{T,N}) where {T, N, P <: Pencil}
        A = typeof(data)
        ndims_space = ndims(pencil)
        E = N - ndims_space
        size_data = size(data)

        extra_dims = ntuple(n -> size_data[n], E)  # = size_data[1:E]
        geom_dims = ntuple(n -> size_data[E + n], ndims_space)  # = size_data[E+1:end]

        if geom_dims !== size_local(pencil)
            throw(DimensionMismatch(
                "Array has incorrect dimensions: $(size_data). " *
                "Local dimensions of pencil: $(size_local(pencil))."))
        end

        new{T, N, A, P, E}(pencil, data, extra_dims)
    end
end

PencilArray(pencil::Pencil, ::Type{T}, extra_dims::Vararg{Int}) where T =
    PencilArray(pencil, Array{T}(undef, extra_dims..., size_local(pencil)...))

PencilArray(pencil::Pencil, extra_dims::Vararg{Int}) =
    PencilArray(pencil, Float64, extra_dims...)

size(x::PencilArray) = size(x.data)

IndexStyle(::PencilArray{T,N,A} where {T,N}) where A = IndexStyle(A)
getindex(x::PencilArray, inds...) = getindex(x.data, inds...)
setindex!(x::PencilArray, v, inds...) = setindex!(x.data, v, inds...)

similar(x::PencilArray, ::Type{S}, dims::Dims) where S =
    PencilArray(x.pencil, similar(x.data, S, dims))

"""
    ndims_extra(x::PencilArray)

Number of "extra" dimensions associated to PencilArray.

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
    gather(x::PencilArray, root::Integer)

Gather data from all MPI processes into one (big) array.

Data is received by the `root` process.

This can be useful for testing, but it shouldn't be used with very large
datasets!
"""
function gather(x::PencilArray, root::Integer)
    comm = p.topology.comm
    rank = MPI.Comm_rank(comm)
    # TODO!!
    nothing
end
