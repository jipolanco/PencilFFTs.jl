module ShiftedArrays

# See https://docs.julialang.org/en/latest/devdocs/offset-arrays

export ShiftedArrayView, has_indices
import Base: @propagate_inbounds

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

function Base.similar(x::ShiftedArrayView, ::Type{S}, dims::Dims) where S
    a = similar(x.data, S, dims)
    ShiftedArrayView(a, x.offsets)
end

function Base.similar(x::ShiftedArrayView{T,N} where T, ::Type{S},
                      inds::NTuple{N, UnitRange{Int}}) where {S,N}
    similar(x, S, length.(inds))
end

# This definition is to avoid type instability.
# By default, instead of 1:1, this function would return Base.OneTo(1) if d > N.
Base.axes(x::ShiftedArrayView{T,N} where T, d) where N =
    d <= N ? axes(x)[d] : 1:1

Base.IndexStyle(::Type{<:ShiftedArrayView{T,N,A}} where {T,N}) where A =
    IndexStyle(A)

# For dimensions N > 1, linear indexing doesn't shift the indices, so that fast
# IndexLinear indexing can still be used when possible.
@propagate_inbounds Base.getindex(x::ShiftedArrayView, i::Int) = x.data[i]

@propagate_inbounds Base.setindex!(x::ShiftedArrayView, v, i::Int) =
    x.data[i] = v

@propagate_inbounds Base.getindex(
        x::ShiftedArrayView{T,N} where T, I::Vararg{Int,N}) where N =
    x.data[(I .- x.offsets)...]

@propagate_inbounds Base.setindex!(
        x::ShiftedArrayView{T,N} where T, v, I::Vararg{Int,N}) where N =
    x.data[(I .- x.offsets)...] = v

# Special case of 1D arrays.
# We always assume that indices are shifted.
Base.IndexStyle(::Type{<:ShiftedArrayView{T,1}} where {T}) = IndexCartesian()

@propagate_inbounds Base.getindex(x::ShiftedArrayView{T,1} where T, i::Int) =
    x.data[i - first(x.offsets)]

@propagate_inbounds Base.setindex!(x::ShiftedArrayView{T,1} where T, v, i::Int) =
    x.data[i - first(x.offsets)] = v

"""
    has_indices(x::ShiftedArrayView, indices...)

Check whether the given set of indices is within the range of a shifted array.
"""
has_indices(x::ShiftedArrayView, I...) = checkbounds(Bool, x, I...)

Base.parent(x::ShiftedArrayView) = parent(x.data)

end  # module ShiftedArrays
