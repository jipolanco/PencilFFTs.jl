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
                   P<:Pencil} <: AbstractArray{T,N}
    pencil :: P
    data   :: A

    function PencilArray(pencil::P,
                         data::AbstractArray{T,N}) where {T, N, P <: Pencil}
        A = typeof(data)
        ndims_space = ndims(pencil)
        ndims_extra = N - ndims_space
        size_data = size(data)
        # Data size, excluding extra (fast) dimensions.
        geom_size = ntuple(n -> size_data[ndims_extra + n], ndims_space)
        if geom_size !== size_local(pencil)
            throw(DimensionMismatch(
                "Array has incorrect dimensions: $(size_data). " *
                "Local dimensions of pencil: $(size_local(pencil))."))
        end
        new{T, N, A, P}(pencil, data)
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
    gather(p::Pencil, x::AbstractArray, root::Integer)

Gather data from all MPI processes into one (big) array.

Data is received by the `root` process.

This can be useful for testing, but it shouldn't be used with very large
datasets!
"""
function gather(x::PencilArray, root::Integer)
    rank = MPI.Comm_rank(p.topology.comm)
    # TODO!!
    nothing
end
