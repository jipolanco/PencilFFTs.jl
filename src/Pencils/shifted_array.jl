# See https://docs.julialang.org/en/latest/devdocs/offset-arrays

"""
    ShiftedArrayView{T,N}

Wraps an array shifting its indices.
"""
struct ShiftedArrayView{T, N,
                        A <: AbstractArray{T,N},
                       } <: AbstractArray{T,N}
    data    :: A
    offsets :: Dims{N}
    axes    :: NTuple{N, UnitRange{Int}}
    function ShiftedArrayView(a::A, offsets::Dims{N}) where
                {T, N, A<:AbstractArray{T,N}}
        ao = map(axes(a), offsets) do ax, off
            ax .+ off
        end
        new{T, N, A}(a, offsets, ao)
    end
end

Base.size(x::ShiftedArrayView) = size(x.data)
Base.axes(x::ShiftedArrayView) = x.axes

# This definition is to avoid type instability.
# By default, instead of 1:1, this function would return Base.OneTo(1) if d > N.
Base.axes(x::ShiftedArrayView{T,N} where T, d) where N =
    d <= N ? axes(x)[d] : 1:1

IndexStyle(::ShiftedArrayView{T,N,A} where {T,N}) where A = IndexStyle(A)

# For dimensions N > 1, linear indexing doesn't shift the indices, so that fast
# IndexLinear indexing can still be used when possible.
Base.getindex(x::ShiftedArrayView, i::Int) = x.data[i]
Base.setindex!(x::ShiftedArrayView, v, i::Int) = x.data[i] = v

Base.getindex(x::ShiftedArrayView{T,N} where T, I::Vararg{Int,N}) where N =
    x.data[(I .- x.offsets)...]
Base.setindex!(x::ShiftedArrayView{T,N} where T, v, I::Vararg{Int,N}) where N =
    x.data[(I .- x.offsets)...] = v

# Special case of 1D arrays.
# We always assume that indices are shifted.
IndexStyle(::ShiftedArrayView{T,1} where {T}) = IndexCartesian()
Base.getindex(x::ShiftedArrayView{T,1} where T, i::Int) =
    x.data[i - first(x.offsets)]
Base.setindex!(x::ShiftedArrayView{T,1} where T, v, i::Int) =
    x.data[i - first(x.offsets)] = v

"""
    data(x::ShiftedArrayView)

Returns array wrapped by a `ShiftedArrayView`.
"""
data(x::ShiftedArrayView) = x.data

"""
    has_indices(x::ShiftedArrayView, indices...)

Check whether the given set of indices is within the range of a shifted array.
"""
has_indices(x::ShiftedArrayView, I...) = checkbounds(Bool, x, I...)

parent(x::ShiftedArrayView) = parent(data(x))

"""
    global_view(x::PencilArray)

Create a [`ShiftedArrayView`](@ref) of a `PencilArray` that takes global
indices.

The order of indices in the returned view is the same as for the original array
`x`. That is, if the indices of `x` are permuted, so are those of the returned
array.
"""
function global_view(x::PencilArray)
    r = range_local(x)
    offsets = first.(r) .- 1
    ShiftedArrayView(x, offsets)
end
